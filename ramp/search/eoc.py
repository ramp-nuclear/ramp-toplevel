"""Tools for reaching the EoC state for the core."""

import logging
from datetime import timedelta
from functools import partial
from math import sqrt
from statistics import NormalDist
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


def find_eoc_from_boc(
    state: OperationalState,
    *,
    rho: PCM,
    regime: Regime,
    at_eoc: Callable[[KResult, PCM], bool],
    too_risky: Callable[[KResult, PCM], bool],
    max_step: Callable,
) -> tuple[OperationalState, OperationalState]:
    """Algorithm to reach eoc, beginning with a BoC core.

    Parameters
    ----------
    state - State at BoC.
    rho - Reactivity to reach at EoC.
    regime - Operational regime to use during this cycle.
    at_eoc - Function to check if we're at EoC.
    too_risky - Function to check if a state is too risky to be used as
                fallback.

    Returns
    -------
    The Bo3 and the EoC state.

    """
    state, kwild, info = _do_first_steps(regime, state)
    if kwild.reactivity < rho and not at_eoc(kwild, rho):
        raise OperationalError(
            "Could not burn the core for the required steps "
            "and without going under the desired reactivity:"
            f" {kwild.reactivity} < {rho}"
        )
    return state, _find_eoc(
        state=state,
        rho=rho,
        regime=regime,
        at_eoc=at_eoc,
        too_risky=too_risky,
        max_step=max_step,
        drhodt=info.drhodt or -100.0,
    )


def _do_first_steps(regime: Regime, state: OperationalState) -> tuple[OperationalState, KResult, BurnResult]:
    info = BurnResult.empty()
    for step in regime.initial_steps:
        state, ninfo = regime.burnstep(state, step)
        info = info + ninfo
    kwild = regime.get_kwild(state)
    return state, kwild, info


def _next_state(
    state: OperationalState,
    *,
    safe: OperationalState,
    regime: Regime,
    rho: PCM,
    drhodt: PCMPerSecond,
    too_risky: Callable[[KResult, PCM], bool],
    max_step: Callable,
    kwild: Optional[KResult] = None,
    safe_reactivity: Optional[PCM] = None,
) -> tuple[OperationalState, OperationalState, PCMPerSecond, Optional[PCM]]:
    kwild = kwild or regime.get_kwild(state)
    forward = kwild.reactivity > rho
    unreliable_drhodt = False
    if (
        safe_reactivity is not None
        and state.history.cycle_time.total_seconds() != safe.history.cycle_time.total_seconds()
    ):
        new_drhodt = (kwild.reactivity - safe_reactivity) / (
            state.history.cycle_time.total_seconds() - safe.history.cycle_time.total_seconds()
        )
        unreliable_new_drhodt = new_drhodt >= 0
        drhodt = drhodt if unreliable_new_drhodt else new_drhodt
    guess = timedelta(seconds=(rho - kwild.reactivity) / drhodt)
    logger.info(f"Stepping from {state.history.cycle_time} with a guess of {guess} reactivity is {kwild.reactivity} ")
    if forward:
        safe = safe if (too_risky(kwild, rho) or unreliable_drhodt) else state
        safe_reactivity = safe_reactivity if (too_risky(kwild, rho) or unreliable_drhodt) else kwild.reactivity
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
        logger.info(f"Doing a step from the safe time {safe.history.cycle_time} to {safe.history.cycle_time + step}")
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


def _find_eoc(
    state: OperationalState,
    *,
    rho: PCM,
    drhodt: PCMPerSecond,
    regime: Regime,
    at_eoc: Callable[[KResult, PCM], bool],
    too_risky: Callable[[KResult, PCM], bool],
    max_step: Callable,
) -> OperationalState:
    safe_state = state
    safe_reactivity = None
    _new_state = partial(
        _next_state,
        rho=rho,
        too_risky=too_risky,
        regime=regime,
        max_step=max_step,
    )
    kwild = regime.get_kwild(state)
    while not at_eoc(kwild, rho):
        state, safe_state, drhodt, safe_reactivity = _new_state(
            state,
            kwild=kwild,
            safe=safe_state,
            drhodt=drhodt,
            safe_reactivity=safe_reactivity,
        )
        kwild = regime.get_kwild(state)
    return state


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


def max_safe_step_at_risk(
    *,
    op_max_timestep: timedelta,
    kres: KResult,
    drhodt: PCMPerSecond,
    rho_target: PCM,
    alpha: float = 1e-8,
    minimal_timestep: timedelta = timedelta(days=1.0),
) -> timedelta:
    """Calculate what is the largest step one can take while likely being able
    to never need to go back, and that all future steps will be long enough
    to not be a problem themselves.

    Assume that at time t=0 we have rho(0) = r0 + N(0, sigma) = N(r0, sigma)
    At time t=t we would have   rho(t) = r0 + t*dr/dt + N(0, sigma)
    However, we do not know r0, so we define instead:
                       z(t) = rho(0) + t*dr/dt + N(0, sigma)
    which is a random variable of the type N(r0 + t*dr/dt, sigma*sqrt(2)).
    We want t such that:    P(z(t) < target) = alpha.
    The previous line is equivalent to:
                P(x < target - rho(0) - t*dr/dt) = alpha
    given that x ~ N(0, sigma*sqrt(2)).
    Using inv_cdf we get:
                P(x < y) = alpha => y = inv_cdf(alpha)
            =>  target - rho(0) - t*dr/dt = y = inv_cdf(alpha).
            =>  t = ((target - y) - rho(0)) / (dr/dt)
    If t exceeds op_max_timestep but subtracting minimal_timestep brings it
    under, we do so — this ensures the following step is at least
    minimal_timestep long (needed for Xe135 equilibrium / drhodt stability).

    Parameters
    ----------
    op_max_timestep - Maximal step we can take between transport operations.
    minimal_timestep - The minimal legal timestep. Around 1 day since Xe135 has
                       to approach equilibrium for drhodt to make sense in the
                       following step.
    kres - The KResult for a current point, used to extrapolate forward.
    drhodt - The burnup worth of the core. Used for extrapolation.
    rho_target - The desired target reactivity value.
    alpha - Probability of falsely saying the step is safe.

    Returns
    -------
    The timestep we can confidently take without violating the rules of conduct.

    """  
    dist = NormalDist(0, kres.reactivity_error * sqrt(2.0))
    y = dist.inv_cdf(alpha)
    tseconds = timedelta(seconds=((rho_target - y) - kres.reactivity) / drhodt)
    if tseconds > op_max_timestep and tseconds - minimal_timestep < op_max_timestep:
        tseconds = tseconds - minimal_timestep
    return min(tseconds, op_max_timestep)


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
    tseconds = timedelta(seconds=(rho_target - kres.reactivity) / drhodt)
    if tseconds > op_max_timestep and tseconds - minimal_timestep < op_max_timestep:
        tseconds = tseconds - minimal_timestep
    return min(tseconds, op_max_timestep)



find_eoc = partial(
    find_eoc_from_boc,
    at_eoc=partial(at_eoc_one_sigma, drho=100.0),
    too_risky=partial(risk_over, alpha=1e-8),
    max_step=max_step_deterministic,
)
