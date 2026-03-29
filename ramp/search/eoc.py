"""Tools for reaching the EoC state for the core."""

import logging
from datetime import timedelta
from functools import partial
from typing import Callable, Optional

import numpy as np
from batman import BurnResult
from corecompute.result import PCM, KResult
from coreoperator.operational_state import OperationalState
from scipy.constants import day

from ramp.regime import Regime
from ramp.utils import PCMPerSecond

logger = logging.getLogger(__name__)


class OperationalError(Exception):
    """An error for when the required operational scheme doesn't make sense."""


def _step(
    *,
    state: OperationalState,
    safe: OperationalState,
    drhodt: PCMPerSecond,
    safe_reactivity: Optional[PCM],
    kwild: KResult,
    rho: PCM,
    regime: Regime,
    too_risky: Callable[[KResult, PCM], bool],
    max_step: Callable,
) -> tuple[OperationalState, OperationalState, PCMPerSecond, Optional[PCM]]:
    """Advance one step toward EOC."""
    forward = kwild.reactivity > rho
    unreliable_drhodt = False
    if (
        safe_reactivity is not None
        and state.history.cycle_time.total_seconds() != safe.history.cycle_time.total_seconds()
    ):
        new_drhodt = (kwild.reactivity - safe_reactivity) / (
            state.history.cycle_time.total_seconds() - safe.history.cycle_time.total_seconds()
        )
        if new_drhodt < 0:
            drhodt = new_drhodt
        else:
            unreliable_drhodt = True
    guess = timedelta(seconds=(rho - kwild.reactivity) / drhodt)
    logger.info(
        f"Stepping from {state.history.cycle_time} with a guess of {guess} reactivity is {kwild.reactivity} "
    )
    if forward:
        skip_update = unreliable_drhodt or too_risky(kwild, rho)
        safe = safe if skip_update else state
        safe_reactivity = safe_reactivity if skip_update else kwild.reactivity
        maxstep = max_step(
            op_max_timestep=regime.maximal_timestep,
            kres=kwild,
            drhodt=drhodt,
            rho_target=rho,
        )
        state, info = regime.burn_to_pcm(state, rho=rho, guess=guess, maxstep=maxstep)
        logger.info(
            f"Burned for {info.time}, \n"
            "Burnup-estimated reactivity at the burned state is "
            f"{info.rho:.0f} PCM\n"
            f"Estimated drho/dt={info.drhodt * day:.0f} PCM/DAY"
        )
    else:
        step = state.history.cycle_time + guess - safe.history.cycle_time
        if step > regime.maximal_timestep:
            step = step / 2
        logger.info(
            f"Doing a step from the safe time {safe.history.cycle_time} to {safe.history.cycle_time + step}"
        )
        if step < timedelta(0):
            raise ValueError(
                f"Computed a negative {step=} for a desired {rho=} and an "
                f"estimated {drhodt=}, with {kwild.reactivity=} and "
                f"{safe_reactivity=}."
                f"\n{guess=}, which should be negative"
                f"\n{state.history.cycle_time=} and {safe.history.cycle_time=}"
            )
        state, info = regime.burnstep(safe, step)
    return state, safe, info.drhodt, safe_reactivity


def find_eoc_from_boc(
    state: OperationalState,
    *,
    rho: PCM,
    regime: Regime,
    at_eoc: Callable[[KResult, PCM], bool],
    too_risky: Callable[[KResult, PCM], bool],
    max_step: Callable,
) -> tuple[OperationalState, OperationalState]:
    """Find EOC starting from a BoC state.

    Parameters
    ----------
    state - State at BoC.
    rho - Reactivity to reach at EoC.
    regime - Operational regime to use during this cycle.
    at_eoc - Predicate that determines if we've reached EOC.
    too_risky - Predicate that determines if the current state is too risky.
    max_step - Callable that computes the maximum allowed timestep.

    Returns
    -------
    The Bo3 and the EoC state.

    """
    # Initial burn steps
    info = BurnResult.empty()
    for step in regime.initial_steps:
        state, ninfo = regime.burnstep(state, step)
        info = info + ninfo
    kwild = regime.get_kwild(state)

    if kwild.reactivity < rho and not at_eoc(kwild, rho):
        raise OperationalError(
            "Could not burn the core for the required steps "
            "and without going under the desired reactivity:"
            f" {kwild.reactivity} < {rho}"
        )

    bo3 = state
    safe = state
    drhodt = info.drhodt or -100.0
    safe_reactivity = None

    while not at_eoc(kwild, rho):
        state, safe, drhodt, safe_reactivity = _step(
            state=state,
            safe=safe,
            drhodt=drhodt,
            safe_reactivity=safe_reactivity,
            kwild=kwild,
            rho=rho,
            regime=regime,
            too_risky=too_risky,
            max_step=max_step,
        )
        kwild = regime.get_kwild(state)

    return bo3, state


def at_eoc_one_sigma(kres: KResult, rho: PCM, drho: PCM) -> bool:
    """Defined being at EOC by having your 1-sigma confidence interval intersect
    the required interval.

    Parameters
    ----------
    kres - KResult for a given state that could be EOC.
    rho - Reactivity to reach at EOC, in PCM.
    drho - Reactivity interval size around rho, in PCM.

    Returns
    -------
    True IFF at eoc according to this intersection.

    """
    return bool(np.isclose(kres.reactivity, rho, atol=drho + kres.reactivity_error))


def risk_over(kres: KResult, rho: PCM, alpha: float) -> bool:
    """Returns true if a normally distributed estimate has too great a risk
    to be under a desired reactivity.

    Parameters
    ----------
    kres - KResult from an Oracle calculation.
    rho - Desired reactivity to reach.
    alpha - Risk you're willing to take. Should be a float in (0,1).

    """
    risk = kres.rho_dist.cdf(rho)
    logger.info(f"The reactivity distribution was: {kres.rho_dist}")
    logger.info(f"The risk was: {risk:.2e}")
    return risk >= alpha


def max_step_deterministic(
    *,
    op_max_timestep: timedelta,
    kres: KResult,
    drhodt: PCMPerSecond,
    rho_target: PCM,
    minimal_timestep: timedelta = timedelta(days=1.0),
) -> timedelta:
    """Calculate the step as the minimum between the time to reach the
    target reactivity and the operational maximum timestep.

    Parameters
    ----------
    op_max_timestep - Maximal step we can take between transport operations.
    kres - The KResult for a current point, used to extrapolate forward.
    drhodt - The burnup worth of the core. Used for extrapolation.
    rho_target - The desired target reactivity value.
    minimal_timestep - Minimum allowed timestep.

    """
    tstep = timedelta(seconds=(rho_target - kres.reactivity) / drhodt)
    if tstep > op_max_timestep and tstep - minimal_timestep < op_max_timestep:
        tstep = tstep - minimal_timestep
    return min(tstep, op_max_timestep)


find_eoc = partial(
    find_eoc_from_boc,
    at_eoc=partial(at_eoc_one_sigma, drho=100.0),
    too_risky=partial(risk_over, alpha=1e-8),
    max_step=max_step_deterministic,
)
