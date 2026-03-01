"""Temperature dependence analysis.

"""
from functools import partial
from pathlib import PurePath
from typing import Tuple, Callable

from coremaker.tree import Node
from coreoperator import OperationalState
from dask import delayed as impure_delayed
from isotopes import U235
from corecompute.query import KQuery

from .util import PCMAndError, diff, DelayedOracleFunc

ValueAndError = Tuple[float, float]
delayed = partial(impure_delayed, pure=True)

kquery = KQuery()

StateChange = Callable[[float], OperationalState]


def _temperature_coefficient(state_factory: StateChange,
                             *, calculator: DelayedOracleFunc, prefix: str = '',
                             hot_temp: float,
                             cold_temp: float,
                             ) -> PCMAndError:
    states = [delayed(state_factory)(temp)
              for temp in (hot_temp, cold_temp)]
    prefixes = (prefix + f'_{name}' for name in ('hot', 'cold'))
    hotres, coldres = [calculator(_state, kquery, prefix=_prefix)
                       for _state, _prefix in zip(states, prefixes)]
    worth = delayed(diff)(hotres, coldres)
    return worth / (hot_temp - cold_temp)


def temperature_coefficient(state: OperationalState, *,
                            calculator: DelayedOracleFunc,
                            prefix: str = '',
                            hot_temperature: float = 40.,
                            cold_temperature: float = 20.
                            ) -> PCMAndError:
    """The temperature coefficient of this core, i.e. what the reactivity change
    is if the core uniformly heats up by 1 degC.

    Calculation assumes a uniform change between uniform temperatures.

    Parameters
    ----------
    state - State to calculate this for. Can be at any temperature.
    calculator - Delayed function that computes the reactivity
    prefix - String used in the beginning of the workspace directory name.
    hot_temperature - Water temperature in the hot state.
    cold_temperature - Water temperature in the cold state.

    Returns
    -------
    Core reactivity's temperature coefficient for the entire core at once,
    including the effects of coolant density change. In PCM/degC.

    """
    return _temperature_coefficient(state.new_temperature,
                                    calculator=calculator, prefix=prefix,
                                    hot_temp=hot_temperature, cold_temp=cold_temperature)


def water_temperature_coefficient(state: OperationalState, *,
                                  calculator: DelayedOracleFunc,
                                  prefix: str = '',
                                  hot_temperature: float = 40.,
                                  cold_temperature: float = 20.
                                  ) -> PCMAndError:
    """The temperature coefficient of water in the core, i.e., what the
    reactivity change is if the water in the core uniformly heats up by 1 degC.

    Calculation assumes a uniform change between uniform water temperatures.
    To ensure this, we set the water in the core at 40 degC and 20 degC.

    Parameters
    ----------
    state - State to calculate this for. Can be at any temperature.
    calculator - Delayed function that computes the reactivity
    prefix - String used in the beginning of the workspace directory name.
    hot_temperature - Water temperature in the hot state.
    cold_temperature - Water temperature in the cold state.

    Returns
    -------
    Water temperature coefficient for the core, in PCM/degC.

    """
    return _temperature_coefficient(state.new_water_temperature,
                                    calculator=calculator, prefix=prefix,
                                    hot_temp=hot_temperature, cold_temp=cold_temperature)


def fuel_temperature_coefficient(state: OperationalState, *,
                                 calculator: DelayedOracleFunc,
                                 prefix: str = '',
                                 hot_temperature: float = 120.,
                                 cold_temperature: float = 20.
                                 ) -> PCMAndError:
    """The temperature coefficient of fuel in the core, i.e., what the
    reactivity change is if the fuel in the core uniformly heats up by 1 degC.

    Calculation assumes a uniform change between uniform fuel temperatures.

    Parameters
    ----------
    state - State to calculate this for. Can be at any temperature.
    calculator - Delayed function that computes the reactivity
    prefix - String used in the beginning of the workspace directory name.
    hot_temperature - Fuel temperature in the hot state.
    cold_temperature - Water temperature in the cold state.

    Returns
    -------
    Water temperature coefficient for the core, in PCM/degC.

    """

    def isfuel(path: PurePath, node: Node):
        return node.mixture and U235 in node.mixture

    return _temperature_coefficient(partial(state.new_temperature, to_change=isfuel),
                                    calculator=calculator, prefix=prefix,
                                    hot_temp=hot_temperature, cold_temp=cold_temperature)
