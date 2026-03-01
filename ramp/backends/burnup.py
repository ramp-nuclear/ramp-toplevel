"""Tools for running the burnup code.

"""
from datetime import timedelta
from typing import Sequence, Tuple, Dict, Optional, Callable

from batman.graphs import DecayGraph, GraphFilter
from batman.solver import InputData, DistEasyData, timestep_constant_power, \
    step_desired_k_at_power, Configuration, BurnResult
from batman.units import Second
from coremaker.protocols.mixture import Mixture
from coreoperator.operational_state import OperationalState
from reactions import Reaction, ReactionRate


BurnupModel = Dict[str, Tuple[DecayGraph, Sequence[Reaction]]]
ReactionModel = Dict[str, Dict[Reaction, ReactionRate]]

NULLFILTER = GraphFilter()


def _partitions_heuristics(n: int):
    """
    A heuristic for the number of partitions in the DistEasyData's dask bag.
    """
    return max(n // 100, 1)


def _pre_exec_burnup(state: OperationalState,
                     burnup_model: BurnupModel,
                     rates: ReactionModel,
                     partition_func: Callable[[int], int]):
    power = state.power_nuc
    named_components = dict(state.core.named_components)
    components = [named_components[name] for name in burnup_model]
    decay_models = [decay_graph for decay_graph, _ in burnup_model.values()]
    reaction_models = [[rates[name][reaction] for reaction in reactions]
                       for name, (_, reactions) in burnup_model.items()]
    volumes = [c.geometry.volume for c in components]
    assert all(vol > 0 for vol in volumes)
    mixtures = [c.mixture for c in components]
    filters = [NULLFILTER for _ in components]
    inputdata = InputData(decay_models, reaction_models, filters, mixtures,
                          volumes)
    data = DistEasyData.from_input(inputdata,
                                   partitions=partition_func(
                                       len(burnup_model)))
    return power, burnup_model.keys(), data


def execute_burnup(*, state: OperationalState,
                   k0: Optional[float] = None,
                   burnup_model: BurnupModel,
                   rates: ReactionModel,
                   time: timedelta,
                   config: Configuration,
                   partition_func: Callable[[int], int]) -> \
        Tuple[Dict[str, Mixture], BurnResult]:
    """Run burnup on some core state given its calculated reaction rates.

    Parameters
    ----------
    state - State to burn.
    k0 - k-eigenvalue of the state at the start of the timestep.
    burnup_model - Model used in the burnup program.
    rates - Calculated ReactionRates.
    time - Time step to step at.
    config - Configuration used in the burnup process.
    partition_func - function used to determine the number of partitions

    """
    power, names, data = _pre_exec_burnup(state=state,
                                          burnup_model=burnup_model,
                                          rates=rates,
                                          partition_func=partition_func)
    mixtures, exinfo = timestep_constant_power(data, power,
                                               time.total_seconds(),
                                               config=config,
                                               k0=k0)
    return dict(zip(names, mixtures)), exinfo


def execute_burnup_to_k(*, state: OperationalState,
                        burnup_model: BurnupModel,
                        rates: ReactionModel, k0: float, k: float,
                        k_tol: float, maxt: timedelta,
                        guess: Optional[timedelta] = None,
                        config: Configuration,
                        partition_func: Callable[[int], int]) -> \
        Tuple[Dict[str, Mixture], BurnResult]:
    """Fire away burnup so it tries to reach some specific k-eigenvalue.

    Parameters
    ----------
    state - Initial state of the system (before time step).
    burnup_model - Burnup model used in the calculation.
    rates - Calculated Reaction Rates for this core state.
    k0 - Initial k-eigenvalue of the system.
    k - Desired k-eigenvalue of the system.
    k_tol - Acceptance tolerance for the k result to be in around k.
    maxt - Maximal time step allowed. Should be around ~2 months.
    guess - A guess for what time step size is needed to reach that condition.
    config - Additional configuration for the burnup process.
    partition_func - function used to determiner the number of partitions

    """
    power, names, data = _pre_exec_burnup(state=state,
                                          burnup_model=burnup_model,
                                          rates=rates,
                                          partition_func=partition_func)
    guess: Optional[Second] = guess and guess.total_seconds()
    mixtures, info = step_desired_k_at_power(
        data, p=power,
        k0=k0, k=k,
        guess=guess,
        k_tolerance=k_tol,
        config=config,
        maxt=maxt.total_seconds())
    return dict(zip(names, mixtures)), info
