from corecompute.query import KQuery
from coreoperator import OperationalState
from dask import delayed
from isotopes import Xe135

from ramp.state_analysis.util import DelayedOracleFunc, PCMAndError, diff


def xenon_worth(
    state: OperationalState,
    *,
    prefix: str = "",
    cold_temperature,
    calculator: DelayedOracleFunc,
) -> PCMAndError:
    cold = state.new_water_temperature(cold_temperature)
    unpoisoned = cold.new_isotope_density({Xe135: 0.0})
    kquery = KQuery()
    cold_result = calculator(cold, kquery, prefix=prefix + "_cold")
    unpoisoned_result = calculator(unpoisoned, kquery, prefix=prefix + "_unpoisoned")

    return delayed(diff)(cold_result, unpoisoned_result)
