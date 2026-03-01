"""The Batman object is in charge of the burnup of core states.

"""
from datetime import timedelta
from typing import Tuple, Optional, Callable, Sequence

from batman import BurnResult, Configuration
from coreoperator.operational_state import OperationalState

from ramp.transport import ReactionScore
from ramp.transport.query import VolumeQuery
from ramp.backends.burnup import BurnupModel, ReactionModel, \
    _partitions_heuristics
from ramp.runners.burnup_runner import run_burnup, run_burnup_to_k
from ramp.transport.result.kresult import PCM
from toolz import groupby

ModelFunc = Callable[[OperationalState], BurnupModel]


class Batman:
    """Object in charge of burnup of core states. It has methods to perform the
    actual burnup process by either taking a specific time step or by burning up
    a certain amount of reactivity. It is also in charge of figuring out which
    reaction queries we need to perform.

    """

    def __init__(self, *, config: Configuration,
                 burnup_model: ModelFunc,
                 partition_func: Callable[[int], int] = _partitions_heuristics,
                 minimal_reactivity_tolerance: PCM = 0):
        """
        Parameters
        ----------
        burnup_model - A strategy that gets the state and returns a decay model and a
                    reaction model for each component in the state's core.
        partition_func - function used to determine the number of burnup cells
                    each worker has to XXXX with, it gets the number of
                    burnup cells and returns the number of cells each worker gets
        minimal_reactivity_tolerance - minimal value for the tolerance when searching
                                       reactivity value, default is 0.
        """
        self.config = config
        self.burnup_model = burnup_model
        self.partition_func = partition_func
        self.minimal_reactivity_tol = minimal_reactivity_tolerance

    def decay_model(self, state: OperationalState) -> BurnupModel:
        """Return the BurnupModel of decaying the core at no power.

        Parameters
        ----------
        state - State to decay

        """
        return {name: (dec, []) for name, (dec, reac) in
                self.burnup_model(state).items()}

    def queries(self, state: OperationalState) -> Sequence[VolumeQuery]:
        """Returns the sequence of reaction queries one must perform to have the
        necessary information for burnup.

        Parameters
        ----------
        state - State to get queries for.

        """
        model = self.burnup_model(state)
        grouped_set = groupby(lambda k: frozenset(model[k][1]), model.keys())
        grouped = {model[seq[0]][1]: seq for seq in grouped_set.values()}
        return tuple(VolumeQuery(tuple(components),
                                 tuple(ReactionScore(reaction, volume_specific=True,
                                                density_specific=True)
                                       for reaction in reactions))
                     for reactions, components in grouped.items())

    def __call__(self, state: OperationalState, *,
                 k0: Optional[float] = None,
                 rates: ReactionModel,
                 time: timedelta) -> Tuple[OperationalState, BurnResult]:
        model = (self.burnup_model(state) if state.power_nuc
                 else self.decay_model(state))
        return run_burnup(state,
                          k0=k0,
                          burnup_model=model,
                          rates=rates,
                          time=time,
                          config=self.config,
                          partition_func=self.partition_func)

    def burn_pcm(self, state: OperationalState, *,
                 rates: ReactionModel,
                 maximal_timestep: timedelta,
                 k: float, drho: PCM, rho_tol: PCM,
                 guess: Optional[timedelta] = None) -> \
            Tuple[OperationalState, BurnResult]:
        """Perform burnup on the state such that the new state has lost a given
        amount of PCM.

        Parameters
        ----------
        state - The operational state to burn.
        rates - The reaction rates queried for this state in each component.
        maximal_timestep - The maximal allowed time step. This is necessary to
                           ensure the calculated rates are still viable.
        k - The k-eigenvalue of the current state.
        drho - The amount of reactivity to burn, in PCM.
        rho_tol - The allowed tolerance around the required value, in PCM.
        guess - Time guess for what time step would be required to hit the goal.

        """
        target = k_target(k, drho)
        return run_burnup_to_k(
            state,
            burnup_model=self.burnup_model(state),
            rates=rates,
            k0=k,
            k=target,
            k_tol=k_tol(target, max(rho_tol, self.minimal_reactivity_tol)),
            maximal_timestep=maximal_timestep,
            config=self.config,
            guess=guess,
            partition_func=self.partition_func)


def k_tol(k: float, rho_tol: PCM) -> float:
    """Calculate the k-eigenvalue tolerance from an initial k and a tolerance
    on the reactivity.

    Parameters
    ----------
    k - k-eigenvalue around the tolerance.
    rho_tol - The reactivity tolerance around k.

    """
    return (k ** 2) * rho_tol * 1e-5


def k_target(k: float, drho: PCM) -> float:
    """Get the target k if original k is given with a desired change in
    reactivity.

    Parameters
    ----------
    k - Original k-eigenvalue
    drho - Desired change in reactivity, in PCM.

    """
    return 1. / (1. / k + drho * 1e-5)
