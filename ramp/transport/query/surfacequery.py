from dataclasses import dataclass
from itertools import pairwise
from pathlib import PurePath
from typing import Tuple

from coremaker.protocols.surface import Surface
from reactions.particle import NamedParticle, Neutron

eV = float


@dataclass(init=True, frozen=True, repr=True)
class SurfaceCurrentQuery:
    """
    represents a query about the current across a surface

    Parameters
    ----------
    surfaces - A surface across whom the current is computed.
    from_components - path of the component from which the current is coming, default is None and then
                      all contributing components are accounted.
    to_components - path of the component to whom the current is going, default is None and then
                      all contributing components are accounted.
    energies - A sequence of incoming energy values such that each successive pair
        defines the bounds of a bin in an energy grid. Should be either empty
        or longer than 2, positive and strictly increasing. Incoming energies
        are given in units of eV.
    particle - Particle type
    """
    surface: Surface
    from_component: PurePath | None = None
    to_component: PurePath | None = None
    energies: Tuple[eV, ...] = ()
    particle: NamedParticle = Neutron

    def __post_init__(self):
        if any(a >= b for a, b in pairwise(self.energies)):
            raise ValueError("The energy grid in a volume query must be strictly "
                             f"increasing. {self.energies} isn't.")
        if any(a < 0. for a in self.energies):
            raise ValueError("The energy grid must be all positive."
                             f" {self.energies[0]}<0, so it failed.")
        if len(self.energies) == 1:
            raise ValueError("An energy grid must have either no elements, which"
                             " means the entire grid, or at least 2. Just one"
                             " is ambiguous about whether we want (0,E) or (E,inf)")
