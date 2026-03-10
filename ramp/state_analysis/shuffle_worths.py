from functools import partial
from itertools import accumulate, chain, pairwise
from operator import itemgetter, matmul
from typing import Any, Callable, Optional, Sequence

from corecompute.query import KQuery
from coreoperator import OperationalState
from coreoperator.mobilization import (
    CyclicShuffle,
    GridAction,
    LoadChain,
    LoadSite,
    Remove,
    Scheme,
)
from dask import delayed
from dask.delayed import Delayed
from more_itertools import last
from multipledispatch import dispatch
from toolz import identity

from ramp.state_analysis.util import OracleFunc, OracleFuncFull, PCMAndError, pcm_err

BasicAction = Remove | LoadSite


@dispatch(Remove, OperationalState)
def decompose_action(action: Remove, state: OperationalState) -> Sequence[BasicAction]:
    return [action]


@dispatch(LoadChain, OperationalState)
def decompose_action(action: LoadChain, state: OperationalState) -> Sequence[BasicAction]:  # noqa
    """
    Decompose a load chain into a sequence of removals and insertions
    according to the minimalistic order of events in the application of a
    load-chain shuffling procedure.
    It is assumed that at any point, only a single site is left empty.
    """
    sites = list(map(itemgetter(0), action.sites))
    transforms = list(map(itemgetter(1), action.sites))
    grid = state.core.grid
    steps = [Remove((last(sites),))]
    for (site, previous_site), transform in zip(pairwise(reversed(sites)), reversed(transforms[1:])):
        rod = grid[previous_site]
        factory = partial(identity, rod)
        factory.__name__ = f"rod_previously_at_{previous_site}"
        steps.append(Remove((previous_site,)))
        steps.append(LoadSite(factory=factory, site=site, transform=transform))
    steps.append(LoadSite(action.factory, sites[0], transforms[0]))
    return steps


@dispatch(CyclicShuffle, OperationalState)
def decompose_action(action: CyclicShuffle, state: OperationalState) -> Sequence[BasicAction]:  # noqa
    raise NotImplementedError(f"decomposing an action is not implementedfor actions of type: {type(action)}")


@dispatch(LoadChain)
def _decomposition_length(action: GridAction):
    return 2 * len(action.sites)


@dispatch(Remove)
def _decomposition_length(action: GridAction):
    return 1


def intermediate_schemes(scheme: Scheme, state: OperationalState, initial: bool = True) -> list[Scheme]:
    """
    Decompose a scheme that defines some shuffling procedure to a series of
    schemes such that each decomposed scheme serves as a model to some
    intermediate stage of the reactor during the application of the scheme.
    Each stage of the reactor is achieved after either the extraction of some
    element from the grid or after the insertion of some element to the grid.
    The order of the sequence of decomposed schemes is chosen to represent
    the actual order of events during the shuffling procedure.
    """
    # noinspection PyTypeChecker
    step_schemes = map(
        lambda x: Scheme((x,)),
        chain(*(decompose_action(action, state) for action in scheme.actions)),
    )
    initial = Scheme() if initial is True else None
    return list(accumulate(step_schemes, func=matmul, initial=initial))


def _intermediate_schemes_length(scheme: Scheme, initial: bool = True):
    return sum(map(_decomposition_length, scheme.actions), start=initial)


def stepwise_shuffle_characteristic(
    state: OperationalState | Delayed,
    scheme: Scheme,
    *,
    characteristic: Callable[[OperationalState], Any],
    initial: bool = True,
) -> dict[Scheme | Delayed, Any]:
    """
    Lazily calculate the characteristic of a state on which a shuffling scheme
    is applied at all of its intermediate stages during the shuffling procedure.
    Each intermediate state is achieved right after the application of some
    "basic" shuffling procedure like extraction of an element from the core's
    grid or the insertion of an element to a specific site in the core's grid.
    This function returns a dictionary between schemes (or lazy schemes)
    (where each scheme defines a snapshot of the reactor's state during
    shuffling) and their corresponding reactivity values.
    For further explanation on the scheme's decomposition process, see
    the documentation of ramp.state_analysis.intermediate_schemes.

    Parameters
    ----------
    state - State to use as a base.
    scheme - A Scheme object that represents a shuffling procedure.
    characteristic - function that computes some characteristic of the state.
        The state is supplied to the function as the first positional argument.
    initial - A boolean to indicate whether to calculate for the input state.
    prefix - String prefix for the workspace directory name.

    Returns
    -------
    A dictionary that maps schemes to (delayed) state characteristic.
    """
    state = state if isinstance(state, Delayed) else delayed(state)
    intermediate_schemes_num = _intermediate_schemes_length(scheme, initial=initial)
    schemes = delayed(intermediate_schemes, nout=intermediate_schemes_num)(scheme, state, initial=initial)

    @delayed
    def _post_shuffle_characteristic(_state: OperationalState, _scheme: Scheme) -> Any:
        _state = _state.new_after_scheme(_scheme)
        return characteristic(_state)

    return {scheme: _post_shuffle_characteristic(state, scheme) for scheme in schemes}


def stepwise_shuffle_reactivity(
    state: OperationalState | Delayed,
    scheme: Scheme,
    *,
    calculator: OracleFuncFull | OracleFunc,
    initial: bool = True,
    prefix: Optional[str] = None,
) -> dict[Scheme, PCMAndError]:
    """
    Lazily calculate the reactivity of a state on which a shuffling scheme is
    applied at all of its intermediate stages during the shuffling procedure.
    Each intermediate state is achieved right after the application of some
    "basic" shuffling procedure like extraction of an element from the core's
    grid or the insertion of an element to a specific site in the core's grid.
    This function returns a dictionary between schemes (where each scheme
    defines a snapshot of the reactor's state during shuffling) and their
    corresponding reactivity values.
    For further explanation on the scheme's decomposition process, see
    the documentation of ramp.state_analysis.intermediate_schemes.

    Parameters
    ----------
    state - State to use as a base.
    scheme - A Scheme object that represents a shuffling procedure.
    calculator - function that computes the reactivity
    initial - A boolean to indicate whether to calculate for the input state.
    prefix - String prefix for the workspace directory name.

    Returns
    -------
    A dictionary that maps schemes to (delayed) state reactivity.
    """

    def _reactivity(_state: OperationalState) -> PCMAndError:
        return pcm_err(calculator(_state, KQuery(), prefix=prefix))

    return stepwise_shuffle_characteristic(state, scheme, characteristic=_reactivity, initial=initial)
