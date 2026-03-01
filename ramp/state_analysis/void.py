"""Void coefficient is defined statically as taking a core state, removing 2%
of its water content, and seeing the reactivity shift, divided by 2.
Therefore, it shows the reactivity swing per percent void.

"""
from functools import partial

from coreoperator import OperationalState
from dask import delayed as impure_delayed
from ramp.state_analysis.util import PCMAndError, diff, DelayedOracleFunc
from corecompute.query import KQuery

delayed = partial(impure_delayed, pure=True)
kquery = KQuery()


def void_coefficient(state: OperationalState, *,
                     calculator: DelayedOracleFunc,
                     void: float = 0.02,
                     prefix: str = ''
                     ) -> PCMAndError:
    """The void coefficient of this core, i.e. what the reactivity change
    is if the core uniformly loses 1% of its water to void.

    Calculation assumes a uniform change between uniform densities.

    Parameters
    ----------
    state - State to calculate this for.
    calculator - Delayed function that computes the reactivity
    void - Fraction of water to turn to void in the calculation.
    prefix - String used in the beginning of the workspace name.

    Returns
    -------
    Core reactivity's void coefficient for the entire core at once, in PCM/%Void.

    """

    states = (state,
              delayed(state.new_water_density_factor)(factor=1.-void))
    prefixes = (prefix + f'_{name}' for name in ('regular', 'voided'))
    refres, voidres = [calculator(_state, kquery, prefix=_prefix)
                       for _state, _prefix in zip(states, prefixes)]
    worth = delayed(diff)(voidres, refres)
    return worth / (100 * void)
