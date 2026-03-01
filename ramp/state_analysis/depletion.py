"""Tools for calculating the depletion information about the core state.

"""
from operator import itemgetter
from pathlib import PurePath
from string import ascii_uppercase as alphabet
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from coremaker.core import Site
from coremaker.protocols.component import Component
from coremaker.protocols.element import Element
from coreoperator import OperationalState
from isotopes import avogadro, Isotope, U235, ZAID
from ramp.state_analysis.util import split_name
from scipy.constants import gram

PerCmBarn = float
kg = float

__all__ = ['maximal_local_depletion',
           'maximal_depletion_plate',
           'maximal_fission_density',
           'average_unloaded_fission_density',
           'average_unloaded_depletion',
           'RodFactory', 'rod_amount', 'print_depletion_map', 'depletion_map']


def _depletion(current: float, initial: float) -> float:
    return 1. - (current / initial)


def maximal_local_depletion(state: OperationalState,
                            isfuel: Callable[[Component], bool],
                            initial_nd: PerCmBarn) -> Tuple[PurePath, float]:
    """The maximal local depletion in any component in the core.

    Parameters
    ----------
    state - State to calculate for.
    isfuel - Function that determines if a component is a fuel component.
    initial_nd - Initial number density of fuel components in the core.

    Returns
    -------
    The name of the most depleted component and its depletion.

    """
    name, min_u235_nd = min(((name, comp.mixture.get(U235))
                             for name, comp in state.core.named_components
                             if isfuel(comp)), key=itemgetter(1))
    return name, _depletion(min_u235_nd, initial_nd)


def maximal_depletion_plate(state: OperationalState,
                            isfuel: Callable[[Component], bool],
                            initial_nd: PerCmBarn,
                            split_name: Callable = split_name
                            ) -> Tuple[str, str, float]:
    """The maximal depletion plate in the core.
    We assume that plate is identified by combination of (site,label)

    Parameters
    ----------
    state - State to calculate for.
    isfuel - Function that determines if a component is a fuel component.
    initial_nd - Initial number density of fuel components in the core.
    split_name - Callable that splits component name into site, label, position

    Returns
    -------
    The site of the most depleted plate it's name and depletion.
    
    """
    df = pd.DataFrame.from_dict(
        {name: {'vol': comp.geometry.volume, 'nd': comp.mixture.get(U235),
                'nd0': initial_nd,
                **dict(zip(('site', 'label', 'x', 'y', 'z'), split_name(name)))}
         for name, comp in state.core.named_components if isfuel(comp)},
        orient='index')
    grouped = df.groupby(['site', 'label']).apply(lambda x: pd.Series(
        {'nd': np.average(x['nd'], weights=x['vol']),
         'nd0': np.average(x['nd0'], weights=x['vol'])}))
    dep = 1.0 - grouped['nd'] / grouped['nd0']
    site, label = dep.idxmax()
    return site, label, dep[site, label]


def maximal_fission_density(state: OperationalState,
                            isfuel: Callable[[Component], bool],
                            total_fission_zaid=ZAID(0, 188, 0),
                            split_name: Callable = split_name
                            ) -> Tuple[str, str, float]:
    """The plate with the maximal fission density in the core.
    We assume that plate is identified by combination of (site,label)

    Parameters
    ----------
    state - State to calculate for.
    isfuel - Function that determines if a component is a fuel component.
    total_fission_zaid - ZAID who accounts for all the fission events
    split_name - Callable that splits component name into site, label, position

    Returns
    -------
    The site of the plate it's name and its fission density.

    """
    df = pd.DataFrame.from_dict(
        {name: {'vol': comp.geometry.volume, 'fission density': comp.mixture.get(total_fission_zaid) * 1e24,
                **dict(zip(('site', 'label', 'x', 'y', 'z'), split_name(name)))}
         for name, comp in state.core.named_components if isfuel(comp)},
        orient='index')
    grouped = df.groupby(['site', 'label']).apply(lambda x: pd.Series(
        {'fission density': np.average(x['fission density'], weights=x['vol'])}))
    fission_density = grouped['fission density']
    site, label = fission_density.idxmax()
    return site, label, fission_density[site, label]


def rod_amount(rod: Element, isotope: Isotope) -> kg:
    """Calculate the amount of an isotope in a given element.

    Parameters
    ----------
    rod - Element to calculate over.
    isotope - Isotope to count.

    """
    return sum(comp.geometry.volume * comp.mixture.get(isotope, 0.) * isotope.mass / avogadro
               for comp in rod.components()
               if not comp.geometry.volume is None) * gram


def _rod_depletion(rod: Element, fresh: Element, isotope: Isotope) -> float:
    amount = rod_amount(rod, isotope)
    fresh = rod_amount(fresh, isotope)
    return _depletion(amount, fresh)


MapAid = Dict[str, Tuple[Isotope, Element]]


def depletion_map(state: OperationalState,
                  freshmap: MapAid
                  ) -> Dict[str, float]:
    """Calculate the depletion map of a given core state.

    Parameters
    ----------
    state - Core state to calculate the map for.
    freshmap - Mapping of sites to the depleting isotope and fresh rod factory.

    Returns
    -------
    The depletion as a fraction in each site data is given for.

    """
    return {site: float(_rod_depletion(state.core.grid[site], fresh, iso))
            for site, (iso, fresh) in freshmap.items()}


def print_depletion_map(state: OperationalState, freshmap: MapAid
                        ) -> None:
    """Print to screen the depletion map of a given core.

    Parameters
    ----------
    state - Core state to print for.
    freshmap - A connection of sites to the depleting isotopes in their rods
               and the factory that generates such a fresh rod.

    """
    depmap = depletion_map(state, freshmap)
    maxlet = max(site[0] for site in freshmap)
    letters = alphabet[:alphabet.index(maxlet) + 1]
    rows = range(1, max(int(site[1:]) for site in freshmap) + 1)
    pmap = [[depmap.get(f'{letter}{row}', '-') for letter in letters]
            for row in rows]
    print('%7s' % '' + ' '.join('%4s   ' % letter for letter in letters))
    for i, row in enumerate(pmap):
        print('%4d   ' % i
              + ' '.join(f'{dep:7.2%}' if isinstance(dep, float) else '   -   '
                         for dep in row))


RodFactory = Callable[[], Element]


def average_unloaded_depletion(state: OperationalState,
                               sites: Iterable[Site],
                               isotope: Isotope,
                               fresh_rod: RodFactory) -> float:
    """The average depletion of the unloaded rods.

    Parameters
    ----------
    state - State to calculate for.
    sites - Sites that are unloaded this cycle.
    isotope - Isotope that we consider depletion with respect to.
    fresh_rod - Factory for generating fresh rods of the type we unload.

    """
    rods = [state.core.grid[site] for site in sites]
    unloaded = sum(rod_amount(rod, isotope) for rod in rods)
    loaded = rod_amount(fresh_rod(), isotope) * len(rods)
    return _depletion(unloaded, loaded)


def average_unloaded_fission_density(state: OperationalState,
                                     sites: Iterable[Site],
                                     isotope: ZAID = ZAID(0, 188, 0)) -> float:
    """The average fission density of the unloaded rods.

    Parameters
    ----------
    state - State to calculate for.
    sites - Sites that are unloaded this cycle.
    isotope - ZAID that accounts for fission events.

    """
    rods = [state.core.grid[site] for site in sites]
    return 1e24 * max(np.average([comp.mixture.get(isotope) for comp in rod.components()
                                  if U235 in comp.mixture],
                                 weights=[comp.geometry.volume for comp in rod.components()
                                          if U235 in comp.mixture]) for rod in rods)
