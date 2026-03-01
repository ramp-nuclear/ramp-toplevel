"""Subpackage that handles and defines how output from a transport
calculation should look like.

"""
from typing import Union, Dict, Tuple

from reactions import ReactionRate

from .kresult import KResult
from .tracksresult import SurfaceTracksResult

EnergyMap = Dict[str, Tuple[float, float]]

Result = Union[KResult, ReactionRate, EnergyMap, Dict]
# TODO: improve type hinting.
