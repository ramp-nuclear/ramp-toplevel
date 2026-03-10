"""Tools to calculate the FSS worth for a core state.

To make things easier to Dask, just assume everything is delayed.

"""

from typing import Container, Iterable, Optional, Sequence

from corecompute.query import KQuery
from coreoperator.operational_state import OperationalState
from dask.delayed import Delayed, delayed

from ramp.regime.controlled_regime import heightwise_characteristic

from .common_states import cold_unpoisoned
from .util import (
    DelayedOracleFunc,
    OracleFunc,
    OracleFuncFull,
    PCMAndError,
    diff,
    pcm_err,
)

degC = float


def all_but(controls: Container[str], rod: str, alias: str) -> bool:
    """A filter that is true if the alias is in the controls but is not a
    specific one.

    """
    return alias in controls and alias != rod


def s_curve(
    state: OperationalState | Delayed,
    alias: str,
    heights: Iterable[float],
    *,
    calculator: OracleFuncFull | OracleFunc,
    prefix: Optional[str] = None,
) -> dict[float, PCMAndError]:
    """Lazily calculate the reactivity of identical states that differ only by the positions of controls.

    Parameters
    ----------
    state - State to use as a base.
    alias - Alias of the controls to move.
    height - Heights at which to position the controls.
    prefix - String prefix for the workspace directory name.
    calculator - function that computes the reactivity

    Returns
    -------
    A dictionary that maps control positions to (delayed) state reactivity.
    """

    def _reactivity(_state):
        return pcm_err(calculator(_state, KQuery(), prefix=prefix))

    return heightwise_characteristic(state, alias, heights, characteristic=_reactivity)


def moveable_reactivity_worth(
    state: OperationalState,
    alias: str,
    extracted: float,
    inserted: float,
    *,
    calculator: OracleFuncFull | OracleFunc,
    prefix: Optional[str] = None,
) -> PCMAndError:
    """Calculate the reactivity worth of moving some movable controls

    Parameters
    ----------
    state - State to use as a base.
    alias - Alias of the controls to move.
    extracted - Extracted height of this control set.
    inserted - Inserted height of this control set.
    prefix - String prefix for the workspace directory name.
    calculator - Delayed function that computes the reactivity

    Returns
    -------
    A tuple of reactivity worth and error, in PCM

    """
    results = s_curve(state, alias, [extracted, inserted], calculator=calculator, prefix=prefix)
    return results[inserted] - results[extracted]


def builtin_reactivity(
    state: OperationalState | Delayed,
    *,
    extracted: float,
    alias: str,
    temperature: degC,
    calculator: DelayedOracleFunc,
    prefix: str = "",
) -> PCMAndError:
    """Calculate the reactivity of the core at its most reactive state

    Parameters
    ----------
    state - State to measure for.
    extracted - Height at which the core is considered extracted.
    alias - Alias to move the rods with.
    temperature - Temperature of the core when it is cold.
    calculator - Delayed function that computes the reactivity
    prefix - String used in the workspace directory.

    """
    extracted = delayed(OperationalState.new_control_height)(state, alias, extracted)
    cold = delayed(cold_unpoisoned)(extracted, temperature)
    kquery = KQuery()
    res = calculator(cold, kquery, prefix=prefix + "_builtin_reactivity")
    return delayed(pcm_err)(res)


def moveable_margin(
    state: OperationalState | Delayed,
    *,
    aliases: Sequence[str],
    all_alias: str,
    inserted: float,
    extracted: float,
    temperature: degC,
    prefix: str = "",
    calculator: OracleFuncFull | OracleFunc,
) -> dict[str, tuple[PCMAndError, PCMAndError]]:
    """Calculate the shutdown margin of movable components.

    Parameters
    ----------
    state - State to calculate for. Does not have to be cold or unpoisoned.
    aliases - Aliases of the movable components.
    all_alias - Alias to move every component.
    inserted - Height that is considered fully inserted.
    extracted - Height that is considered fully extracted.
    temperature - The cold temperature of the core.
    prefix - String used in the beginning of the workspace directory's name.
    calculator - Delayed function that computes the reactivity

    Returns
    -------
    dictionary between component alias and its corresponding
    shutdown margin (without taking inaccuracies into account).

    """
    extracted = delayed(OperationalState.new_control_height)(state, alias=all_alias, height=extracted)
    cold = delayed(cold_unpoisoned)(extracted, temperature)
    insert = delayed(OperationalState.new_control_height)
    inserted_cold_unpoisoned = [insert(cold, alias=alias, height=inserted) for alias in aliases]

    @delayed
    def _calculator(_state, _prefix: str = ""):
        # noinspection PyArgumentList
        return calculator(_state, KQuery(), prefix=_prefix)

    ins_results = [
        _calculator(_state, _prefix=prefix + f"_cold_unpoisoned_{alias}_inserted")
        for alias, _state in zip(aliases, inserted_cold_unpoisoned)
    ]
    ext_result = _calculator(cold, _prefix=prefix + "_cold_unpoisoned_extracted")

    return {
        alias: (-delayed(pcm_err)(ins_result), delayed(diff)(ext_result, ins_result))
        for alias, ins_result in zip(aliases, ins_results)
    }


def bank_margin_no_uncertainties(
    state: OperationalState,
    *,
    aliases: Sequence[str],
    all_alias: str,
    inserted: float,
    extracted: float,
    temperature: degC,
    calculator: DelayedOracleFunc,
    prefix: str = "",
) -> tuple[PCMAndError, PCMAndError, str]:
    """The shutdown margin of the entire control rod bank. This is defined as the minimal
    margin of all possible N-1 rod configurations in the most reactive state of the core.

    Parameters
    ----------
    state - Initial core state. Does not have to be the most reactive.
    aliases - Aliases for the different rod configurations.
    all_alias - Alias to move every component.
    inserted - Height that is considered fully inserted.
    extracted - Height that is considered fully extracted.
    temperature - Core temperature when cold.
    calculator - Delayed function that computes the reactivity
    prefix - String used in the beginning of the workspace directory's name.

    Returns
    -------
    The margin and its error, in PCM, not accounting for inaccuracies.

    """

    def _min_margin(
        m: dict[str, tuple[PCMAndError, PCMAndError]],
    ) -> tuple[PCMAndError, PCMAndError, str]:
        alias, (margin, worth) = min(m.items(), key=lambda x: x[1][0])
        return margin, worth, alias

    margins = moveable_margin(
        state,
        aliases=aliases,
        inserted=inserted,
        extracted=extracted,
        temperature=temperature,
        calculator=calculator,
        all_alias=all_alias,
        prefix=prefix,
    )
    return delayed(_min_margin, nout=3)(margins)


def bank_margin(
    state: OperationalState,
    *,
    aliases: Sequence[str],
    all_alias: str,
    inserted: float,
    extracted: float,
    temperature: degC,
    calculator: OracleFunc | OracleFuncFull,
    worth_uncertainty: float,
    reactivity_uncertainty=0,
    prefix: str = "",
) -> tuple[PCMAndError, PCMAndError, str, PCMAndError]:
    """The same as bank_margin_no_uncertainties but taking the bank reactivity uncertainties into account.

    Parameters
    ----------
    state - Initial core state. Does not have to be the most reactive.
    aliases - Aliases for the different rod configurations.
    all_alias - Alias to move every component.
    inserted - Height that is considered fully inserted.
    extracted - Height that is considered fully extracted.
    temperature - Core temperature when cold.
    calculator - Delayed function that computes the reactivity
    worth_uncertainty - Administrative uncertainty in the FSS worth.
    prefix - String used in the beginning of the workspace directory's name.

    See Also
    --------
    bank_margin_no_uncertainties

    Returns
    -------
    The FSS margin and the computational error associated with it.

    """
    as_is, worth, alias = bank_margin_no_uncertainties(
        state,
        aliases=aliases,
        all_alias=all_alias,
        inserted=inserted,
        extracted=extracted,
        temperature=temperature,
        calculator=calculator,
        prefix=prefix,
    )
    return (
        as_is - worth * worth_uncertainty - reactivity_uncertainty,
        worth,
        alias,
        as_is,
    )
