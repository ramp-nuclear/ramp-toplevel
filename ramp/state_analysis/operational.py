"""Operational related quantities.

"""
from datetime import timedelta
from functools import partial

from coreoperator import OperationalState
from dask import delayed as unpure_delayed
from scipy.constants import day

from ramp.state_analysis.util import OracleFunc, OracleFuncFull, diff
from ramp.transport import KQuery

MWday = float
delayed = partial(unpure_delayed, pure=True)
PCM = float
PCM_MWday = float


def cycle_time_length(boc: OperationalState,
                      eoc: OperationalState) -> timedelta:
    return eoc.history.timedelta - boc.history.timedelta


def cycle_length(boc: OperationalState, eoc: OperationalState) -> MWday:
    dt = cycle_time_length(boc, eoc).total_seconds() / day
    return dt * boc.power_nuc

