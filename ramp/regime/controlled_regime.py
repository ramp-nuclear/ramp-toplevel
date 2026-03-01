"""An operational regime where the core is kept critical using a control system.

"""
import logging
from copy import deepcopy
from datetime import timedelta
from operator import itemgetter
from pathlib import PurePath
from typing import Sequence, Tuple, ContextManager, Any, Iterable, Callable

from batman import BurnResult
from coreoperator.operational_state import OperationalState
from cytoolz import compose
from dask import delayed, compute
from dask.delayed import Delayed
from ramp.backends.burnup import ReactionModel
from ramp.batman import Batman
from corecompute.oracle import Oracle
from ramp.regime.regime import Regime, reaction_rates_calculator
from corecompute.result import KResult, PCM
from ramp.utils.docs import append_doc_of
from reactions import Reaction, ReactionRate
from toolz import curry

logger = logging.getLogger(__name__)
DEFAULT_TOLERANCE = 3e3
delayed = curry(delayed, pure=True)


@curry
def data_is_relevant(data, tolerance: PCM) -> Tuple[KResult, Any]:
    """
    function removes the reaction rates data if the state is far from critical,
    as no burnup will be preformed in this state.
    """
    if abs(data[0].reactivity) < tolerance:
        return data
    return data[0], None


def heightwise_characteristic(state: OperationalState | Delayed,
                              alias: str,
                              heights: Iterable[float], *,
                              characteristic: Callable[[OperationalState], Any]) \
        -> dict[float, Delayed | Any]:
    state = state if isinstance(state, Delayed) else delayed(state)

    def _characteristic(_state: OperationalState, height: float) -> Any:
        _state = _state.new_control_height(alias=alias, height=height)
        # noinspection PyArgumentList
        return characteristic(_state)

    if hasattr(characteristic, '__name__'):
        _characteristic.__name__ = characteristic.__name__
    return {height: delayed(_characteristic)(state, height)
            for height in heights}


def critical_height(results_per_height: dict[float, [KResult, dict[PurePath, dict[Reaction, ReactionRate]]]]) -> float:
    crit_height, _ = min(
        ((height, abs(results[0].reactivity))
         for height, results in results_per_height.items()),
        key=itemgetter(1))
    return crit_height


class ControlledRegime(Regime):

    """A regime, only this one ensures the core is critical when doing burnup by moving rods.

    """

    def __init__(self, *, maximal_timestep: timedelta,
                 initial_steps: Sequence[timedelta], oracle: Oracle,
                 batman: Batman, control_alias,
                 guesses: Sequence[float],
                 outside: float,
                 tolerance: PCM = DEFAULT_TOLERANCE,
                 oracle_context: ContextManager = None,
                 batman_context: ContextManager = None,
                 ):
        super().__init__(maximal_timestep=maximal_timestep,
                         initial_steps=initial_steps, oracle=oracle,
                         batman=batman,
                         oracle_context=oracle_context,
                         batman_context=batman_context)
        self.control_alias = control_alias
        self.guesses = guesses
        self.outside = outside
        self.heights = set(guesses)
        self.heights.add(outside)
        self.tolerance = tolerance
        self.result_per_height = {}
        self.last_history_name = None

    def _calc_state(self, state: OperationalState, **oracle_kwargs) -> \
            Tuple[KResult, ReactionModel]:
        if self.to_update(state):
            self.update_results_per_height(state, **oracle_kwargs)
        height = state.history.current_params[self.control_alias]
        if height not in self.result_per_height:
            height = critical_height(self.result_per_height)
        return self.result_per_height[height]

    @append_doc_of(Regime.burnstep)
    def burnstep(self, state: OperationalState,
                 step: timedelta, **oracle_kwargs) -> Tuple[OperationalState, BurnResult]:
        new_state, info = super().burnstep(state, step, **oracle_kwargs)
        return self.get_controlled_state(new_state), info

    @append_doc_of(Regime.burn_to_pcm)
    def burn_to_pcm(self, state: OperationalState, **kwargs) -> \
            Tuple[OperationalState, BurnResult]:
        new_state, info = super().burn_to_pcm(state, **kwargs)
        return self.get_controlled_state(new_state), info

    def to_update(self, state: OperationalState) -> bool:
        """
        method that determines when the reactions rates and reactivity should be updated.

        Parameters
        ----------
        state: OperationalState
         The new state

        Returns
        -------
        bool
        """
        if self.last_history_name is None:
            return True
        new_history = deepcopy(state.history)
        new_history.append({self.control_alias: self.outside})
        if (new_history, state.design_name) != self.last_history_name:
            return True
        return False
    # noinspection PyMissingOrEmptyDocstring

    @append_doc_of(Regime.get_kwild)
    def get_kwild(self, state: OperationalState) -> KResult:
        if self.to_update(state):
            self.update_results_per_height(state)
        kwild = self.result_per_height[self.outside][0]
        logger.info(
            f'At {self._time_since_boc(state)} since boc, '
            f'the wild reactivity is {kwild.reactivity}.'
        )
        return kwild
    # noinspection PyMissingOrEmptyDocstring

    @append_doc_of(Regime.get_controlled_state)
    def get_controlled_state(self, state: OperationalState) -> OperationalState:
        if self.to_update(state):
            self.update_results_per_height(state)
        crit_height = critical_height(self.result_per_height)
        logger.info(f'Critical height is {crit_height}')
        return state.new_control_height(self.control_alias,
                                        crit_height)

    def update_results_per_height(self, state: OperationalState, **oracle_kwargs) -> None:
        """
        return controlled state (state with control rods inside).
        used for correcting control height during burnup
        """
        logger.info(f'Updating reaction rates and reactivity in '
                    f'different heights for {state=}')
        with self.oracle_context():
            _reaction_rates_calculator = curry(reaction_rates_calculator,
                                               direct=self.oracle,
                                               burnup_queries_factory=self.batman.queries,
                                               **oracle_kwargs)
            relevant_reaction_rates = compose(data_is_relevant(tolerance=self.tolerance),
                                              _reaction_rates_calculator)
            self.result_per_height, = compute(heightwise_characteristic(state, self.control_alias,
                                                                        self.heights, characteristic=relevant_reaction_rates))
        if all(val is None
               for val in map(itemgetter(1), self.result_per_height.values())):
            raise ValueError(f'The reactivity of {state} was not within '
                             f'{self.tolerance} pcm of criticality at all heights.')
        new_history = deepcopy(state.history)
        new_history.append({self.control_alias: self.outside})
        self.last_history_name = (new_history, state.design_name)


