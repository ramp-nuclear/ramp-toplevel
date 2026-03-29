"""The Batman object is in charge of the burnup of core states."""

from datetime import timedelta
from pathlib import PurePath
from typing import Callable, Iterable, Optional, Sequence

from batman import BurnResult, Configuration
from batman.graphs import DecayGraph, GraphFilter
from batman.solver import DistEasyData, InputData, step_desired_k_at_power, timestep_constant_power
from batman.units import Second
from corecompute.query import ReactionScore, VolumeQuery
from corecompute.result import PCM
from coreoperator.operational_state import OperationalState
from reactions import Reaction, ReactionRate
from toolz import groupby

BurnupModel = dict[PurePath, tuple[DecayGraph, Sequence[Reaction]]]
ReactionModel = dict[PurePath, dict[Reaction, ReactionRate]]


def batman_partitions_heuristics(n: int):
    """A heuristic for the number of partitions in the DistEasyData's dask bag."""
    return max(n // 100, 1)

NULLFILTER = GraphFilter()

ModelFunc = Callable[[OperationalState], BurnupModel]


class Batman:
    """Object in charge of burnup of core states. It has methods to perform the
    actual burnup process by either taking a specific time step or by burning up
    a certain amount of reactivity. It is also in charge of figuring out which
    reaction queries we need to perform.

    """

    def __init__(
        self,
        *,
        config: Configuration,
        burnup_model: ModelFunc,
        partition_func: Callable[[int], int] = batman_partitions_heuristics,
        minimal_reactivity_tolerance: PCM = 0,
    ):
        """
        Parameters
        ----------
        burnup_model - A strategy that gets the state and returns a decay model and a
                    reaction model for each component in the state's core.
        partition_func - function used to determine the number of burnup cells
                    each worker has to deal with, it gets the number of
                    burnup cells and returns the number of cells each worker gets
        minimal_reactivity_tolerance - minimal value for the tolerance when searching
                                       reactivity value, default is 0.
        """
        self.config = config
        self.burnup_model = burnup_model
        self.partition_func = partition_func
        self.minimal_reactivity_tol = minimal_reactivity_tolerance

    def _prepare(
        self,
        state: OperationalState,
        burnup_model: BurnupModel,
        rates: ReactionModel,
    ) -> tuple[float, Iterable[PurePath], DistEasyData]:
        power = state.power_nuc
        named_components = dict(state.core.named_components)
        components = [named_components[name] for name in burnup_model]
        decay_models = [decay_graph for decay_graph, _ in burnup_model.values()]
        reaction_models = [
            [rates[name][reaction] for reaction in reactions] for name, (_, reactions) in burnup_model.items()
        ]
        volumes = [c.geometry.volume for c in components]
        assert all(vol > 0 for vol in volumes)
        mixtures = [c.mixture for c in components]
        filters = [NULLFILTER for _ in components]
        inputdata = InputData(decay_models, reaction_models, filters, mixtures, volumes)
        data = DistEasyData.from_input(inputdata, partitions=self.partition_func(len(burnup_model)))
        return power, burnup_model.keys(), data

    def decay_model(self, state: OperationalState) -> BurnupModel:
        """Return the BurnupModel of decaying the core at no power.

        Parameters
        ----------
        state - State to decay

        """
        return {name: (dec, []) for name, (dec, reac) in self.burnup_model(state).items()}

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
        return tuple(
            VolumeQuery(
                tuple(components),
                tuple(ReactionScore(reaction, volume_specific=True, density_specific=True) for reaction in reactions),
            )
            for reactions, components in grouped.items()
        )

    def __call__(
        self,
        state: OperationalState,
        *,
        k0: Optional[float] = None,
        rates: ReactionModel,
        time: timedelta,
    ) -> tuple[OperationalState, BurnResult]:
        model = self.burnup_model(state) if state.power_nuc else self.decay_model(state)
        power, names, data = self._prepare(state, model, rates)
        mixtures, info = timestep_constant_power(data, power, time.total_seconds(), config=self.config, k0=k0)
        return state.burnup(mixtures=dict(zip(names, mixtures)), time=time), info

    def burn_pcm(
        self,
        state: OperationalState,
        *,
        rates: ReactionModel,
        maximal_timestep: timedelta,
        k: float,
        drho: PCM,
        rho_tol: PCM,
        guess: Optional[timedelta] = None,
    ) -> tuple[OperationalState, BurnResult]:
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
        tol = k_tol(target, max(rho_tol, self.minimal_reactivity_tol))
        power, names, data = self._prepare(state, self.burnup_model(state), rates)
        guess_seconds: Optional[Second] = guess and guess.total_seconds()
        mixtures, info = step_desired_k_at_power(
            data,
            p=power,
            k0=k,
            k=target,
            guess=guess_seconds,
            k_tolerance=tol,
            config=self.config,
            maxt=maximal_timestep.total_seconds(),
        )
        return state.burnup(mixtures=dict(zip(names, mixtures)), time=info.time), info


def k_tol(k: float, rho_tol: PCM) -> float:
    """Calculate the k-eigenvalue tolerance from an initial k and a tolerance
    on the reactivity.

    Parameters
    ----------
    k - k-eigenvalue around the tolerance.
    rho_tol - The reactivity tolerance around k.

    """
    return (k**2) * rho_tol * 1e-5


def k_target(k: float, drho: PCM) -> float:
    """Get the target k if original k is given with a desired change in
    reactivity.

    Parameters
    ----------
    k - Original k-eigenvalue
    drho - Desired change in reactivity, in PCM.

    """
    return 1.0 / (1.0 / k + drho * 1e-5)
