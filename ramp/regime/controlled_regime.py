"""An operational regime where the core is kept critical using a control system."""

import logging
from datetime import timedelta
from functools import wraps
from operator import itemgetter
from pathlib import PurePath
from typing import Any, Callable, ContextManager, Iterable, Sequence

from corecompute.oracle import Oracle
from corecompute.result import PCM, KResult
from coreoperator.operational_state import OperationalState
from dask import compute, delayed
from dask.delayed import Delayed
from reactions import Reaction, ReactionRate
from toolz import compose, curry

from ramp.batman import Batman, ReactionModel
from ramp.regime.regime import Regime, reaction_rates_calculator
from ramp.utils.docs import append_doc_of

logger = logging.getLogger(__name__)
DEFAULT_TOLERANCE = 3e3
delayed = curry(delayed, pure=True)


@curry
def data_is_relevant(data, tolerance: PCM) -> tuple[KResult, Any]:
    """
    function removes the reaction rates data if the state is far from critical,
    as no burnup will be preformed in this state.

    This saves on memory in the cache.

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

    dchar = delayed(_characteristic)
    return {height: dchar(state, height) for height in heights}


def critical_height(
    results_per_height: dict[float, [KResult, dict[PurePath, dict[Reaction, ReactionRate]]]],
) -> float:
    crit_height, _ = min(
        ((height, abs(results[0].reactivity)) for height, results in results_per_height.items()),
        key=itemgetter(1),
    )
    return crit_height


def _key(state: OperationalState):
    return (
        ("history_data", (state.history.cycles, state.history.cycle_time, state.history.cycle_burnup)),
        ("tags", frozenset(state.tags)),
    )


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
        self.result_per_height: dict[Any, dict[float, tuple[KResult, ReactionModel]]] = {}

    def clear_cache_except(self, state: OperationalState | None = None) -> None:
        """Clears the result_per_height cache except for the one which matches the state.

        The ReactionModel results can be pretty memory intensive, so clearing things can
        be beneficial. We clear things by default when time changes are performed.

        """
        key = _key(state) if state is not None else None
        self.result_per_height = {key: self.result_per_height[key]} if key and key in self.result_per_height else {}

    def _calc_state(
        self,
        state: OperationalState,
        strict_height: bool = False,
        **oracle_kwargs,
    ) -> tuple[KResult, ReactionModel]:
        self.update_results_per_height(state, **oracle_kwargs)
        key = _key(state)
        height = state.params[self.control_alias] if strict_height else critical_height(self.result_per_height[key])
        return self.result_per_height[key][height]

    @append_doc_of(Regime.get_kwild)
    def get_kwild(self, state: OperationalState) -> KResult:
        self.update_results_per_height(state)
        kwild = self.result_per_height[_key(state)][self.outside][0]
        logger.info(f"At {self._time_since_boc(state)} since boc, the wild reactivity is {kwild.reactivity}.")
        return kwild

    @append_doc_of(Regime.get_controlled_state)
    def get_controlled_state(self, state: OperationalState) -> OperationalState:
        self.update_results_per_height(state)
        crit_height = critical_height(self.result_per_height[_key(state)])
        fraction = (crit_height - min(self.heights)) / (max(self.heights) - min(self.heights))
        logger.info(f"Critical height is {crit_height}, which is {fraction:.1%} of the control range")
        return state.new_control_height(self.control_alias, crit_height)

    def update_results_per_height(self, state: OperationalState, force: bool = False, **oracle_kwargs) -> None:
        """
        return controlled state (state with control rods inside).
        used for correcting control height during burnup
        """
        key = _key(state)
        if not force and key in self.result_per_height:
            return
        logger.info(f"Updating reaction rates and reactivity in different heights for {state=}")
        with self.oracle_context():
            _reaction_rates_calculator = curry(
                reaction_rates_calculator,
                direct=self.oracle,
                burnup_queries_factory=self.batman.queries,
                **oracle_kwargs,
            )
            relevant_reaction_rates = compose(data_is_relevant(tolerance=self.tolerance), _reaction_rates_calculator)
            (self.result_per_height[key],) = compute(
                heightwise_characteristic(
                    state,
                    self.control_alias,
                    self.heights,
                    characteristic=relevant_reaction_rates,
                )
            )
        if all(val is None for val in map(itemgetter(1), self.result_per_height[key].values())):
            raise ValueError(
                f"The reactivity of {state} was not within {self.tolerance} pcm of criticality at all heights."
            )


def _cache_clear(method):
    @wraps(method)
    def _wrap(self: ControlledRegime, *args, clear_cache: bool = False, **kwargs):
        new_state, ex = method(self, *args, **kwargs)
        if clear_cache:
            self.clear_cache_except(new_state)
        return self.get_controlled_state(new_state), ex

    return _wrap


for method_name in ["burnstep", "decay", "burn_to_pcm"]:
    setattr(ControlledRegime, method_name, _cache_clear(getattr(Regime, method_name)))
