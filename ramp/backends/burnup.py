"""Tools for running the burnup code."""

from typing import Sequence

from batman.graphs import DecayGraph
from reactions import Reaction, ReactionRate

BurnupModel = dict[str, tuple[DecayGraph, Sequence[Reaction]]]
ReactionModel = dict[str, dict[Reaction, ReactionRate]]


def batman_partitions_heuristics(n: int):
    """
    A heuristic for the number of partitions in the DistEasyData's dask bag.
    """
    return max(n // 100, 1)
