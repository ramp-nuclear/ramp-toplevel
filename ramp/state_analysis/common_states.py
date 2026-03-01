"""Tools to make common modifications to states.

"""
from datetime import timedelta
from functools import partial
from itertools import repeat, chain, starmap
from pathlib import PurePath
from typing import Dict, Optional, Callable, Sequence, Iterable

import numpy as np
from coremaker.core import Core, TREE_NAME
from coremaker.materials import Mixture
from coremaker.protocols.geometry import Geometry
from coremaker.transform import Transform, identity
from coremaker.tree import Tree, Node, ChildType
from coreoperator.operational_state import OperationalState
from dask import delayed
from dask.delayed import Delayed
from isotopes import Xe135, ZAID
from scipy.constants import day

from ramp.oracle import Oracle
from ramp.regime import Regime
from ramp.regime.controlled_regime import heightwise_characteristic
from ramp.search.eoc import at_eoc_one_sigma, risk_over, max_safe_step_at_risk, \
    find_eoc_from_boc
from ramp.state_analysis.util import invert, pcm_err
from ramp.transport import KQuery

degC = float
ND = float


def unpoisoned(state: OperationalState,
               unpoison: Optional[Dict[ZAID, ND]] = None) -> OperationalState:
    unpoison = unpoison or {Xe135: 0.}
    return state.new_isotope_density(unpoison)


def cold_unpoisoned(state: OperationalState,
                    temperature: degC = 20.,
                    unpoison: Optional[Dict[ZAID, ND]] = None
                    ) -> OperationalState:
    cold = state.new_water_temperature(temperature)
    return unpoisoned(cold, unpoison)


def eoc_from_boc(state: OperationalState, regime: Regime) -> OperationalState:
    at_eoc = partial(at_eoc_one_sigma, drho=100.)
    risky = partial(risk_over, alpha=1e-8)
    safe_step = partial(max_safe_step_at_risk, alpha=1e-8)
    find_eoc = partial(find_eoc_from_boc,
                       at_eoc=at_eoc,
                       too_risky=risky,
                       max_safe_step=safe_step)
    return find_eoc(state, regime=regime, rho=0.)


def _divide_period(period: float, resolution: float) -> Iterable[timedelta]:
    split, rest = divmod(period, resolution)
    yield from repeat(timedelta(days=resolution), int(split))
    if rest > 0.:
        yield timedelta(days=rest)


def divide_periods(periods: Iterable[float], resolutions: Iterable[float]) -> \
        Sequence[timedelta]:
    """
    Tool to divide the cycle operation time into time intervals of a specified
    resolution.

    Example
    ---------
    divide a period of 8 days to intervals of 1 day length in the first
    3 days and intervals of 2 day length in the last 5 days.
    >>> times = divide_periods([3, 5], [1, 2])
    >>> [time.total_seconds() / day for time in times]
    [1.0, 1.0, 1.0, 2.0, 2.0, 1.0]
    """
    return list(chain.from_iterable(starmap(_divide_period,
                                            zip(periods, resolutions))))


def midcycle_states(state: OperationalState,
                    regime: Regime, times: Sequence[timedelta],
                    saver: Callable[[str, OperationalState], None],
                    initially_critical: bool = True,
                    save_boc: bool = True,
                    t0: timedelta = timedelta(0.),
                    **oracle_kwargs) -> OperationalState:
    """
    Generate operational states in intermediate stages of a cycle's operation.
    The operational states are "saved" using the input "saver".
    The first and final states are saved by the keys ('boc', 'eoc')
    correspondingly. Each intermediate key is the total time of burnup of the
    corresponding state since the beginning of cycle in days (as a string
    with a single digit after the floating point).
    This keying method assumes that each supplied timedelta is larger
    than 0.1 days.

    Parameters
    ----------
    state - The boc state.
    regime - A ramp.regime.regime.Regime object that performs burnup on
        operational states.
    times - A sequence of datetime.timedelta objects to define the intermediate
        steps during operation in which the corresponding reactor state is
        saved. It is assumed that the sum of all the input times is the
        cycle's period.
    saver - The operational states are "saved" using the input "saver". A
        sensible shelve based saver that dumps the states as pickles to the
        file system is easily constructed using the
        ramp.state_analysis.util.shelve_saver tool.
        saver is a callable that receives a key (as a string) and an operational
        state.
    initially_critical - A boolean to specify whether to compute the critical
        control heights of the input state.
    save_boc - A boolean to specify whether to save the boc state using
        the saver. Usually False when restarting a burnup run.
    t0 - A timedelta to shift the times that are used as keys to the saver.
        useful for restarting a simulation from a state which is not the BOC.
    oracle_kwargs - keyword arguments that are supplied to the Regime.burnup
        method.
    """
    if not initially_critical:
        state = regime.get_controlled_state(state)
    if save_boc:
        saver('boc', state)
    ti = state.history.timedelta
    for time in times[:-1]:
        state, _ = regime.burnup(state, time, **oracle_kwargs)
        saver(f'{(state.history.timedelta - ti + t0).total_seconds() / day:.1f}',
              state)
    state, _ = regime.burnup(state, times[-1], **oracle_kwargs)
    saver('eoc', state)
    return state


def _change_height_and_compute(state, alias, height, oracle) -> float:
    query = KQuery()
    return oracle(state.new_control_height(alias, height), query)[query][0].reactivity


def find_critical(state: OperationalState, guesses: Sequence[float], oracle: Oracle,
                  alias: str = "FSS") -> Delayed:
    """
    function to find the critical height of a state using parallel reactivity computations at different heights
    and linear interpolation to find the critical height.

    Parameters
    ----------
    state: OperationalState
    guesses: Sequence[float]
     The heights at which to compute th e reactivity
    oracle: Oracle
     The oracle to be used for the reactivity computation.
    alias: str
     alias for the shutdown system.

    Returns
    -------
    Delayed of an OperationalState
    """

    def reactivity(_state):
        return pcm_err(oracle(_state, KQuery())).nominal_value

    heightwise_reactivity = list(heightwise_characteristic(state, alias, guesses,
                                                           characteristic=reactivity).values())
    height = delayed(invert)(0, guesses, heightwise_reactivity)
    return delayed(state.new_control_height)(alias, height)


def critical_heights_temp_xenon_variations(state: OperationalState, height_search_length: float,
                                           guess_num: int, oracle: Oracle, alias: str = "FSS",
                                           cold_temp=20.) -> dict[
    str, OperationalState]:
    """
    function that get a state in HFP with control rods in critical position and finds the states at
    CZP and HZP in with control rods in critical position, returning a dict of the name of the state
    and the OperationalState

    Parameters
    ----------
    state: OperationalState
     State at HFP, with control rods in critical position
    height_search_length: float
     The length to the lowest search point for the critical height
    guess_num: int
     The number of search points
    oracle: Oracle
     The oracle to compute the reactivity
    alias: str
     Alias to move the control_rods
    cold_temp: degC
     The temperature at cold state

    Returns
    -------
     dict[str, OperationalState]
     Dict that contains 3 states as values.
    """
    HFP_height = state.history.current_params[alias]
    guesses = np.linspace(HFP_height - height_search_length,
                          HFP_height, guess_num)

    no_xenon = delayed(state.new_isotope_density)({Xe135: 0})
    HZP = find_critical(no_xenon, guesses, oracle, alias)
    CZP = find_critical(delayed(no_xenon.new_temperature)(cold_temp), guesses, oracle, alias)
    return dict(HFP=state, HZP=HZP, CZP=CZP)


ENCOMPASSING_PATH = PurePath('outer_region')


def _realias(alias: PurePath) -> PurePath:
    if alias.parents[-2] == TREE_NAME:
        return TREE_NAME / ENCOMPASSING_PATH / alias.relative_to(TREE_NAME)
    else:
        return alias


# Maybe unused?
def encompassed(state: OperationalState, mixture: Mixture,
                encompassing_geometry: Geometry,
                transform: Transform = identity) -> OperationalState:
    """
    Generate a new OperationalState by encompassing an existing one with
    the specified geometry filled with the specified mixture
    """
    tree = Tree()
    tree.nodes[ENCOMPASSING_PATH] = Node(encompassing_geometry,
                                         mixture=mixture,
                                         transform=transform)
    new_core: Core = state.new_core()
    new_core._outer_geometry = encompassing_geometry
    tree.graft(new_core.tree, ENCOMPASSING_PATH, ChildType.exclusive)
    new_core.tree = tree
    new_core.aliases = {name: (alias[0], list(map(_realias, alias[1])))
                        for name, alias in new_core.aliases.items()}
    return state.copy(core=new_core)
