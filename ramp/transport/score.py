from dataclasses import dataclass
from typing import Sequence, Protocol

from numpy.typing import ArrayLike
from ramp.utils.grid import linearize, thin
from reactions import ReactionType


@dataclass(init=True, frozen=True)
class Score:
    """
    score
    Parameters
    ----------
    name: name of the score
    volume specific : True -> the score is per volume (cm**3)
    """
    name: str
    volume_specific: bool


class TabulatedValues(Protocol):
    x: Sequence[float]

    def __call__(self, x: float) -> float:
        ...


@dataclass(init=True, frozen=True)
class TabulatedScore:
    """
    tabulated score with linear-linear interpolation
    Parameters
    ----------
    energy: energy grid in eV
    score: Array of scores matching energy grid
    volume specific : True -> the score is per volume (cm**3)
    """
    energy: ArrayLike
    score: ArrayLike
    volume_specific: bool

    def __post_init__(self):
        if len(self.energy) <= 1:
            raise ValueError('length of energy must be >=2')
        if len(self.score) <= 1:
            raise ValueError('length of score must be >=2')
        if len(self.energy) != len(self.score):
            raise ValueError('lengths of score and energy must match')

    @classmethod
    def from_tabulated1d(cls, tabulated: TabulatedValues, volume_specific: bool):
        energy, score = thin(*linearize(tabulated.x, tabulated))
        return cls(energy, score, volume_specific)


@dataclass(init=True, frozen=True)
class ReactionScore:
    """
    reaction score
    Parameters
    ----------
    reaction: reaction type
    volume_specific: True -> the score is per volume (cm**3)
    density_specific: True -> the score is per number density (1/(barn cm))
    """
    reaction: ReactionType
    volume_specific: bool
    density_specific: bool
