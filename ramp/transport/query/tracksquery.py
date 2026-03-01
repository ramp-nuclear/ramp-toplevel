from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from coremaker.protocols.surface import Surface
from reactions import Particle


@dataclass(init=True, frozen=True)
class SurfaceTracksQuery:
    path: Path
    surfaces: tuple[Surface, ...]
    particles: tuple[Particle, ...]
    maximal_number_of_particles: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, 'path', self.path.absolute())
