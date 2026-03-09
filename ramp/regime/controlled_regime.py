"""An operational regime where the core is kept critical using a control system."""

import logging
from datetime import timedelta
from operator import itemgetter
from pathlib import PurePath
from typing import Sequence, ContextManager, Any, Iterable, Callable

from batman import BurnResult
from coreoperator.operational_state import OperationalState
from toolz import compose, curry
from dask import delayed, compute
from dask.delayed import Delayed
from ramp.backends.burnup import ReactionModel
from ramp.batman import Batman
from corecompute.oracle import Oracle
from ramp.regime.regime import Regime, reaction_rates_calculator
from corecompute.result import KResult, PCM
from ramp.utils.docs import append_doc_of
from reactions import Reaction, ReactionRate

logger = logging.getLogger(__name__)
DEFAULT_TOLERANCE = 3e3
delayed = curry(delayed, pure=True)


@curry
def data_is_relevant(data, tolerance: PCM) -> tuple[KResult, Any]:
    """
    function removes the reaction rates data if the state is far from critical,
    as no burnup will be preformed in this state.
    """
    if abs(data[0].reactivity) < tolerance:
        return data
    return data[0], None


def heightwise_characteristic(
    state: OperationalState | Delayed,
    alias: str,
    heights: Iterable[float],
    *,
    characteristic: Callable[[OperationalState], Any],
) -> dict[float, Delayed | Any]:
    state = state if isinstance(state, Delayed) else delayed(state)

    def _characteristic(_state: OperationalState, height: float) -> Any:
        _state = _state.new_control_height(alias=alias, height=height)
        return characteristic(_state)

    if hasattr(characteristic, "__name__"):
        _characteristic.__name__ = characteristic.__name__
    return {height: delayed(_characteristic)(state, height) for height in heights}


def critical_height(
    results_per_height: dict[
        float, [KResult, dict[PurePath, dict[Reaction, ReactionRate]]]
    ],
) -> float:
    crit_height, _ = min(
        (
            (height, abs(results[0].reactivity))
            for height, results in results_per_height.items()
        ),
        key=itemgetter(1),
    )
    return crit_height


class ControlledRegime(Regime):
    """A regime, only this one ensures the core is critical when doing burnup."""

    def __init__(
        self,
        *,
        maximal_timestep: timedelta,
        initial_steps: Sequence[timedelta],
        oracle: Oracle,
        batman: Batman,
        control_alias,
        guesses: Sequence[float],
        outside: float,
        tolerance: PCM = DEFAULT_TOLERANCE,
        oracle_context: ContextManager = None,
        batman_context: ContextManager = None,
    ):
        super().__init__(
            maximal_timestep=maximal_timestep,
            initial_steps=initial_steps,
            oracle=oracle,
            batman=batman,
            oracle_context=oracle_context,
            batman_context=batman_context,
        )
        self.control_alias = control_alias
        self.guesses = guesses
        self.outside = outside
        self.heights = set(guesses)
        self.heights.add(outside)
        self.tolerance = tolerance
        self.result_per_height = {}

    def _calc_state(
        self,
        state: OperationalState,
        **oracle_kwargs,
    ) -> tuple[KResult, ReactionModel]:
        self.update_results_per_height(state, **oracle_kwargs)
        height = state.params[self.control_alias]
        if height not in self.result_per_height:
            height = critical_height(self.result_per_height)
        return self.result_per_height[height]

    @append_doc_of(Regime.burnstep)
    def burnstep(
        self, state: OperationalState, step: timedelta, **oracle_kwargs
    ) -> tuple[OperationalState, BurnResult]:
        new_state, info = super().burnstep(state, step, **oracle_kwargs)
        return self.get_controlled_state(new_state), info

    @append_doc_of(Regime.burn_to_pcm)
    def burn_to_pcm(
        self, state: OperationalState, **kwargs
    ) -> tuple[OperationalState, BurnResult]:
        new_state, info = super().burn_to_pcm(state, **kwargs)
        return self.get_controlled_state(new_state), info

    @append_doc_of(Regime.get_kwild)
    def get_kwild(self, state: OperationalState) -> KResult:
        self.update_results_per_height(state)
        kwild = self.result_per_height[self.outside][0]
        logger.info(
            f"At {self._time_since_boc(state)} since boc, "
            f"the wild reactivity is {kwild.reactivity}."
        )
        return kwild

    @append_doc_of(Regime.get_controlled_state)
    def get_controlled_state(self, state: OperationalState) -> OperationalState:
        self.update_results_per_height(state)
        crit_height = critical_height(self.result_per_height)
        fraction = (crit_height - min(self.heights)) / (
            max(self.heights) - min(self.heights)
        )
        logger.info(
            f"Critical height is {crit_height}, which is {fraction:.1%} of the control range"
        )
        return state.new_control_height(self.control_alias, crit_height)

    def update_results_per_height(
        self, state: OperationalState, **oracle_kwargs
    ) -> None:
        """
        return controlled state (state with control rods inside).
        used for correcting control height during burnup
        """
        logger.info(
            f"Updating reaction rates and reactivity in different heights for {state=}"
        )
        with self.oracle_context():
            _reaction_rates_calculator = curry(
                reaction_rates_calculator,
                direct=self.oracle,
                burnup_queries_factory=self.batman.queries,
                **oracle_kwargs,
            )
            relevant_reaction_rates = compose(
                data_is_relevant(tolerance=self.tolerance), _reaction_rates_calculator
            )
            (self.result_per_height,) = compute(
                heightwise_characteristic(
                    state,
                    self.control_alias,
                    self.heights,
                    characteristic=relevant_reaction_rates,
                )
            )
        if all(
            val is None for val in map(itemgetter(1), self.result_per_height.values())
        ):
            raise ValueError(
                f"The reactivity of {state} was not within "
                f"{self.tolerance} pcm of criticality at all heights."
            )
