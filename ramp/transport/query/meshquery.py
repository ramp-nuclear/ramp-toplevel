from dataclasses import dataclass
from itertools import pairwise

from coremaker.mesh import CartesianMesh, CylindricalMesh, SphericalMesh
from coremaker.transform import Transform, identity
from reactions import Particle, Neutron

from ramp.transport import Score

eV = float

Mesh = CartesianMesh | CylindricalMesh | SphericalMesh


@dataclass(init=True, frozen=True)
class MeshQuery:
    """
    represents a mesh query about some volume averaged quantity.

    Parameters
    ----------
    mesh - mesh to calculate flux at.
    energies - An energy grid. Should be either empty or longer than 2, positive
               and strictly increasing. Energies given in eV.
    transform - A transform of the mesh relative
    particle - particle to query data about
    """
    mesh: Mesh
    scores: tuple[Score, ...] = (Score('flux', volume_specific=True),)
    energies: tuple[eV, ...] = ()
    transform: Transform = identity
    particle: Particle = Neutron

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
