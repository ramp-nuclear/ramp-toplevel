"""Find the critical control rod height."""

from functools import partial
from typing import Callable

from coreoperator.operational_state import OperationalState
from corecompute.query import KQuery
from corecompute.oracle import Oracle
from ramp.search.choice import best_choice
from corecompute.result import KResult, PCM


def find_height_at_given_reactivity(
    oracle: Oracle, state: OperationalState, alias, guesses, given_reactivity: PCM = 0.0
):
    """Find the best height out of a list of guesses for the core to be at
        the reactivity specified by {given_reactivity} [pcm]

    Parameters
    ----------
    oracle - Oracle to use to get the reactivity.
    state - Original state to perturb.
    alias - Control alias to change to move the control objects.
    guesses - Guesses to find the best from.
    given_reactivity - the given reactivity for which to find a corresponding
        state

    Returns
    -------
    3-tuple of the critical height, the state at that height and the KResult
    that proves it.

    """
    factory = partial(state.new_control_height, alias)
    scoring = partial(_scoring, func=oracle.direct, given_reactivity=given_reactivity)
    # noinspection PyTypeChecker
    return best_choice(factory, scoring, guesses)


def find_critical_height(
    oracle: Oracle, state: OperationalState, alias, guesses
) -> tuple[float, OperationalState, KResult]:
    """Find the best height out of a list of guesses for the core to be critical

    Parameters
    ----------
    oracle - Oracle to use to get the reactivity.
    state - Original state to perturb.
    alias - Control alias to change to move the control objects.
    guesses - Guesses to find the best from.

    Returns
    -------
    3-tuple of the critical height, the state at that height and the KResult
    that proves it.

    """
    return find_height_at_given_reactivity(
        oracle=oracle, state=state, alias=alias, guesses=guesses, given_reactivity=0.0
    )


def _scoring(
    x: OperationalState, func: Callable, given_reactivity
) -> tuple[float, KResult]:
    query = KQuery()
    (kresult,) = func(x, query)[query]
    return -abs(kresult.rho - given_reactivity), kresult
