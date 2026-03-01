"""Regime is an almost god-like object that handles how the reactor is operated
during the cycle. It's in charge of doing burnup at the correct specifications.

"""
import itertools as it
import logging
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from functools import lru_cache
from pathlib import PurePath
from typing import Sequence, Tuple, Optional, Callable, ContextManager

from batman import BurnResult
from coreoperator.operational_state import OperationalState
from dask import compute, delayed
from more_itertools import difference
from ramp.backends.burnup import ReactionModel
from ramp.batman import Batman
from corecompute.oracle import Oracle
from corecompute.query import VolumeQuery, ReactionScore, KQuery
from corecompute.result import PCM, KResult
from reactions import Reaction, ReactionRate

logger = logging.getLogger(__name__)

kquery = KQuery()


class Regime:
    """The object that handles the operational regime for the reactor.
    It can be used for calculating the effect of the reactor operation on
    its material content, i.e perform burnup simulations for a given
    reactor state.

    Parameters
    ----------
    maximal_timestep - The maximal time (in time units) between two
        following reaction rate calculations.
    initial_steps - Specified times during operation (in time units) in
        which the reaction rates are calculated.
    oracle - The tool with which reaction rates are calculated.
    batman - The tool with which burnup calculations are performed
    oracle_context - A callable to a context inside which oracle calculations
        are performed. A likely use case is for sending oracle calculation tasks
        directly to a specific dask scheduler.
    batman_context - A callable to a context inside which batman calculations
        are performed. A likely use case is for sending batman calculation tasks
        directly to a specific dask scheduler.
    """

    def __init__(self, *, maximal_timestep: timedelta,
                 initial_steps: Sequence[timedelta],
                 oracle: Oracle,
                 batman: Batman,
                 oracle_context: Optional[Callable[[], ContextManager]] = None,
                 batman_context: Optional[Callable[[], ContextManager]] = None):
        self.maximal_timestep = maximal_timestep
        self.initial_steps = initial_steps
        self.oracle = oracle
        self.batman = batman
        self.oracle_context = oracle_context or nullcontext
        self.batman_context = batman_context or nullcontext

    @staticmethod
    def _time_since_boc(state: OperationalState):
        return state.history.cycle_time

    def burnup(self, state: OperationalState,
               time: timedelta, **oracle_kwargs) -> Tuple[
        OperationalState, BurnResult]:
        """Burn the core up a known amount of time.

        Parameters
        ----------
        state - Operational state to burn.
        time - Time interval to burn.
        oracle_kwargs - keyword arguments that are passed down to the oracle

        """
        t0 = self._time_since_boc(state)
        info = BurnResult.empty()
        for step in self._steps(t0, t0 + time):
            state, new_info = self.burnstep(state, step, **oracle_kwargs)
            info = info + new_info
        return state, info

    def burnstep(self, state: OperationalState,
                 step: timedelta, **oracle_kwargs) -> Tuple[
        OperationalState, BurnResult]:
        """Perform a single burnup step of a known time step.

        Parameters
        ----------
        state - State to burn.
        step - Exact step to take.
        oracle_kwargs - keyword arguments that are passed down to the oracle

        """
        if step > self.maximal_timestep:
            raise ValueError(f'{step=} is larger than '
                             f'maximal_timestep={self.maximal_timestep}.')
        if step < timedelta(0):
            raise ValueError(f'{step=} must be positive')
        kresult, rates = self._calc_state(state, **oracle_kwargs)
        with self.batman_context():
            return self.batman(state, rates=rates, time=step, k0=kresult.k)

    def decay(self, state: OperationalState,
              time: timedelta) -> Tuple[OperationalState, BurnResult]:
        """Decay the core at 0 power.

        Parameters
        ----------
        state - State to decay from
        time - Length of decay period.

        """
        if power := state.power_nuc:
            raise ValueError(f'decay step is illegal because '
                             f'{power=}')
        if time < timedelta(0):
            raise ValueError(f'{time=} must be positive')
        with self.batman_context():
            return self.batman(state, rates={}, time=time)

    def burn_to_pcm(self, state: OperationalState, *, rho: PCM,
                    guess: Optional[timedelta] = None,
                    maxstep: Optional[timedelta] = None, **oracle_kwargs) -> \
            Tuple[OperationalState, BurnResult]:
        """Burn the core so it reaches a specific reactivity.

        Parameters
        ----------
        state - State to burn.
        rho - Desired reactivity to reach, in PCM.
        guess - A guess for what time step it would take to reach this rho.
        maxstep - Maximal allowed time step to take in this guess.
        oracle_kwargs - keyword arguments that are passed down to the oracle

        """
        if guess and guess < timedelta(0):
            raise ValueError(f'{guess=} must be positive')
        maxstep = maxstep or self.maximal_timestep
        kwild = self.get_kwild(state)
        drho = kwild.reactivity - rho
        logger.info(f'trying to burn {drho} PCM.')
        kresult, rates = self._calc_state(state, **oracle_kwargs)
        with self.batman_context():
            return self.batman.burn_pcm(state, rates=rates,
                                        maximal_timestep=maxstep,
                                        k=kresult.k,
                                        drho=drho,
                                        rho_tol=kresult.reactivity_error,
                                        guess=guess)

    @lru_cache(maxsize=4)
    def get_kwild(self, state: OperationalState) -> KResult:
        """Return the KResult for the state at its least compensated state.

        """
        kwild, _ = self._calc_state(state)
        logger.info(
            f'At {self._time_since_boc(state)} since boc, '
            f'the wild reactivity is {kwild.reactivity}.'
        )
        return kwild

    def _calc_state(self, state: OperationalState, **oracle_kwargs) -> \
            Tuple[KResult, ReactionModel]:
        logger.info(f'Computing reaction rates of {state}')
        with self.oracle_context():
            results, = compute(
                delayed(reaction_rates_calculator)(state, self.oracle, self.batman.queries,
                                                   **oracle_kwargs))
        return results

    def _steps(self, t0: timedelta, t: timedelta) -> Sequence[timedelta]:
        if t <= t0:
            raise ValueError(f'{t=} must be greater than {t0=}.')
        times = it.accumulate(self.initial_steps)
        times = it.dropwhile(lambda x: x <= t0, times)
        times = it.dropwhile(lambda x: x >= t, times)
        times = [t0, *times]
        results = list(difference(times, initial=False))
        t0 = max(times)
        # noinspection PyTypeChecker
        n, rest = divmod(t - t0, self.maximal_timestep)
        results.extend(it.repeat(self.maximal_timestep, times=n))
        results.append(rest)
        return [step for step in results if step.total_seconds() > 0]

    def get_controlled_state(self,
                             state: OperationalState) -> OperationalState:
        """Return the state at its most critical state by compensation.

        Parameters
        ----------
        state - State to start the guesswork with.

        """
        return state


def reaction_rates_calculator(state: OperationalState, direct: callable,
                              burnup_queries_factory: Callable[[OperationalState], Sequence[VolumeQuery]],
                              **oracle_kwargs) -> [KResult, dict[PurePath,
                                                                 dict[Reaction,
                                                                      ReactionRate]]]:
    burnup_queries = burnup_queries_factory(state)
    results = direct(state, *burnup_queries, kquery, **oracle_kwargs)
    rates = defaultdict(dict)
    for rate in it.chain.from_iterable(results.values()):
        match rate:
            case {'component': component,
                  'score': ReactionScore(reaction=reaction,
                                         volume_specific=True,
                                         density_specific=True),
                  'value': value,
                  'error': error,
                  'particle': 'neutron',
                  **kwargs} if not kwargs:
                rates[component][reaction] = ReactionRate(component, reaction,
                                                          value, error)

    return results[kquery][0], dict(rates)
