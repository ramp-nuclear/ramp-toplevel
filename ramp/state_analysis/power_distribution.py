"""Tools for analyzing the power distribution in a core state."""

from functools import partial

import numpy as np
import pandas as pd
from coreoperator import OperationalState
from dask import delayed as unpure_delayed
from isotopes import U235
from ramp.state_analysis.control_rod_worth import DelayedOracleFunc
from ramp.state_analysis.util import split_name
from corecompute.query import KQuery, VolumeQuery
from corecompute.query.score import Score, SUPPORTED_SCORE
from corecompute.result import EnergyMap, KResult

delayed = partial(unpure_delayed, pure=True)
__all__ = ["ppf", "ppf_and_k", "power_map"]


def _analyze_pff(power_map: dict[str, tuple[float, float]]) -> tuple[float, float, str]:
    powers = np.array(list(power_map.values()))
    keys = list(power_map.keys())
    mean_pow = np.mean(powers, axis=0).item()
    max_pow, max_pow_err = np.max(powers, axis=0)
    _ppf = max_pow / mean_pow
    maxloc = np.argmax(powers, axis=0).item()
    max_name = keys[maxloc]
    return _ppf, max_pow_err / mean_pow, max_name


def _analyze_radial_pff(
    power_map: dict[str, tuple[float, float]],
) -> tuple[str, str, float]:
    df = pd.DataFrame.from_dict(
        {
            name: {
                "power": value[0],
                **dict(zip(("site", "label", "x", "y", "z"), split_name(name))),
            }
            for name, value in power_map.items()
        },
        orient="index",
    )
    grouped = df.groupby(["site", "label"]).apply(
        lambda x: pd.Series({"power": np.average(x["power"])})
    )
    power = grouped["power"]
    site, label = power.idxmax()
    return site, label, power[site, label] / power.mean()


def ppf(
    state: OperationalState, *, calculator: DelayedOracleFunc, prefix: str = ""
) -> tuple[tuple[float, float, str], tuple[str, str, float]]:
    """Get the Power Peaking Factor and the radial Power Peaking Factor of a core at a given state.

    Parameters
    ----------
    state - State to calculate in.
    calculator - Delayed function with the signature of the "Oracle" protocol.
    prefix - String to use in the beginning of the workspace directory.

    Returns
    -------
    The PPF, the standard deviation in said value, and the name of the component
    with the maximal energy deposition value.
    Also the Radial ppf, the site and the plate number where it is obtained.

    """
    pmap = power_map(state, calculator=calculator, prefix=prefix)[0]
    return delayed(_analyze_pff)(pmap), delayed(_analyze_radial_pff)(pmap)


def ppf_and_k(
    state: OperationalState, *, calculator: DelayedOracleFunc, prefix: str = ""
) -> tuple[tuple[float, float, str], KResult]:
    """Get the Power Peaking Factor and the multiplication factor of a core at a given state.

    Parameters
    ----------
    state - State to calculate in.
    calculator - Delayed function with the signature of the "Oracle" protocol.
    prefix - String to use in the beginning of the workspace directory.

    Returns
    -------
    tuple of the PPF and the k.
    The PPF is given with the standard deviation in said value, and the name of the component
    with the maximal energy deposition value.

    """
    pmap, kresult = power_map(state, calculator=calculator, prefix=prefix)
    return delayed(_analyze_pff)(pmap), kresult


def power_map(
    state,
    *,
    calculator: DelayedOracleFunc,
    prefix,
    power_computation_score: SUPPORTED_SCORE = "fission-q-prompt",
) -> tuple[EnergyMap, KResult]:
    """Get the power map of a core at a given state and the k.

    Parameters
    ----------
    state - State to calculate in.
    calculator - Delayed function with the signature of the "Oracle" protocol.
    prefix - String to use in the beginning of the workspace directory.
    power_computation_score - name of Score to be used to compute the power, default is fission-q-prompt.

    Returns
    -------
    The powermap and the k

    """

    @delayed
    def _query(state):
        return VolumeQuery(
            tuple(
                name for name, cmp in state.core.named_components if U235 in cmp.mixture
            ),
            scores=(Score(power_computation_score, volume_specific=True),),
        )

    ppfquery = _query(state)
    kquery = KQuery()
    results = calculator(state, ppfquery, kquery, prefix=prefix + "_ppf")

    def _postprocess(results):
        return {x.component: (x.value, x.error) for x in results}

    return delayed(_postprocess)(results[ppfquery]), results[kquery][0]
