"""Operational related quantities."""

from datetime import timedelta
from functools import partial

from coreoperator import OperationalState
from dask import delayed as unpure_delayed

MWday = float
delayed = partial(unpure_delayed, pure=True)
PCM = float
PCM_MWday = float


def cycle_time_length(eoc: OperationalState) -> timedelta:
    return eoc.history.cycle_time


def cycle_length(eoc: OperationalState) -> MWday:
    return eoc.history.cycle_burnup
