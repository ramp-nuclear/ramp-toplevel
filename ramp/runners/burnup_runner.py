"""Tools used to run burnup. Should be merged with the backends/burnup.

"""
from datetime import timedelta
from typing import Tuple, Optional, Callable

from coreoperator.operational_state import OperationalState
from batman import BurnResult, Configuration
from ramp.backends.burnup import execute_burnup, BurnupModel, \
    execute_burnup_to_k, ReactionModel


def run_burnup(state: OperationalState, *,
               k0: Optional[float] = None,
               burnup_model: BurnupModel,
               rates: ReactionModel,
               time: timedelta,
               config: Configuration,
               partition_func: Callable[[int], int]) -> Tuple[
    OperationalState, BurnResult]:
    mixtures, exinfo = execute_burnup(state=state, burnup_model=burnup_model,
                                      rates=rates, k0=k0,
                                      time=time, config=config,
                                      partition_func=partition_func)
    return state.burnup(mixtures=mixtures, time=time), exinfo


def run_burnup_to_k(state: OperationalState, *,
                    burnup_model: BurnupModel,
                    rates: ReactionModel,
                    guess: timedelta = None,
                    k0: float, k: float, k_tol: float,
                    maximal_timestep: timedelta,
                    config: Configuration,
                    partition_func: Callable[[int], int]) -> \
        Tuple[OperationalState, BurnResult]:
    mixtures, exinfo = execute_burnup_to_k(state=state,
                                           burnup_model=burnup_model,
                                           rates=rates,
                                           guess=guess,
                                           k_tol=k_tol,
                                           k0=k0, k=k,
                                           config=config,
                                           maxt=maximal_timestep,
                                           partition_func=partition_func)
    return state.burnup(mixtures=mixtures, time=exinfo.time), exinfo
