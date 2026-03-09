from operator import itemgetter
from typing import Iterable

from coremaker.protocols.grid import Site
from coreoperator import OperationalState
from coreoperator.mobilization import Scheme, Remove
from dask import delayed
from dask.delayed import Delayed
from corecompute.oracle import OracleResult
from ramp.state_analysis.util import PCMAndError, diff, OracleFuncFull, OracleFunc
from corecompute.query import KQuery


def rods_extraction_worth(
    state: OperationalState | Delayed,
    sites=Iterable[Site],
    *,
    calculator: OracleFuncFull | OracleFunc,
    prefix: str = "",
) -> dict[Site, PCMAndError]:
    """
    calculate the worth of the individual extraction of each of the rods
    which are specified by their corresponding sites in the core's grid.
    state - The state from which the rods are extracted.
    sites - An iterable of sites that defines the group of rods for which the
        extraction worth is calculated.
    calculator - function that computes the reactivity
    prefix - String used in the workspace directory.
    """
    prefix = prefix + "_rod_worths"
    kquery = KQuery()

    @delayed
    def calc_reactivity_after_removal(_state, site) -> OracleResult:
        removal_scheme = Scheme(actions=(Remove(sites=[site]),))
        _state = _state.new_after_scheme(removal_scheme)
        return calculator(_state, kquery, prefix=prefix)

    state = state if isinstance(state, Delayed) else delayed(state)
    delayed_all_inserted_kresult = delayed(calculator)(state, kquery, prefix=prefix)
    delayed_rodless_kresults = {
        site: calc_reactivity_after_removal(state, site) for site in sites
    }
    delayed_diff = delayed(diff)
    return {
        site: delayed_diff(delayed_rodless_kresult, delayed_all_inserted_kresult)
        for site, delayed_rodless_kresult in delayed_rodless_kresults.items()
    }


def maximal_rod_worth(
    state: OperationalState,
    sites=Iterable[Site],
    *,
    calculator: OracleFuncFull | OracleFunc,
    prefix: str = "",
) -> tuple[Site, PCMAndError]:
    """
    finds the site whose extraction has maximal worth among the given sites.
    state - The state from which the rods are extracted.
    sites - An iterable of sites that defines the group of rods for which the
        extraction worth is calculated.
    calculator - function that computes the reactivity
    prefix - String used in the workspace directory.
    """
    rod_worths = rods_extraction_worth(
        state, sites, calculator=calculator, prefix=prefix
    )

    def _max_worth(m: dict[Site, PCMAndError]) -> tuple[Site, PCMAndError]:
        site, worth = max(m.items(), key=itemgetter(1))
        return site, worth

    return delayed(_max_worth, nout=2)(rod_worths)
