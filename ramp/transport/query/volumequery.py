""""Queries that are averaged over the volume of a component

"""
from dataclasses import dataclass
from pathlib import PurePath
from typing import Tuple, Sequence

from more_itertools import pairwise
from ramp.transport.score import Score, ReactionScore
from reactions import Neutron
from reactions.particle import NamedParticle

eV = float


@dataclass(init=True, frozen=True, repr=True)
class VolumeQuery:
    """
    represents a query about some volume averaged quantity.

    Parameters
    ----------
    names - A tuple of component names to tally at.
    scores - A sequence of things to tally at the component volumes.
                 Usually these  would be reaction rates, or the default flux score.
    energies - A sequence of incoming energy values such that each successive pair
        defines the bounds of a bin in an energy grid. Should be either empty
        or longer than 2, positive and strictly increasing. Incoming energies
        are given in units of eV.
    particle - Particle type
    """
    names: Tuple[PurePath, ...]
    scores: Sequence[Score | ReactionScore] = (Score('flux', volume_specific=True),)
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
