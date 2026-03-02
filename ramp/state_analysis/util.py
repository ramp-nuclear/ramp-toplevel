"""Analysis utilities.

"""
import re
from abc import abstractmethod
from functools import partial
from pathlib import PurePath
from typing import Protocol, Union, Tuple, Optional, Callable, Sequence

import uncertainties
from coreoperator import OperationalState
from toolz import compose, juxt
from dask import delayed as unpure_delayed
from numpy import floor, log10, negative, interp
from corecompute.oracle import OracleResult
from corecompute.result import KResult
from corecompute.query import KQuery, Query
from uncertainties import ufloat, std_dev, nominal_value

__all__ = ['PCMAndError', 'pcm_err', 'diff', 'split_name', 'invert', 'sround', 'decompose_ufloats',
           'OracleFunc', 'OracleFuncFull', 'DelayedOracleFunc']


_Floatlike = Union[float, "PCMAndError"]


class PCMAndError(Protocol):
    """A protocol that mimics uncertainties' UFloat. For some reason they add
    their methods after class declaration so their operators don't appear in
    type checks.

    """

    @abstractmethod
    def __neg__(self) -> "PCMAndError": pass

    @abstractmethod
    def __add__(self, other: _Floatlike) -> "PCMAndError": pass

    @abstractmethod
    def __radd__(self, other: _Floatlike) -> "PCMAndError": pass

    @abstractmethod
    def __sub__(self, other: _Floatlike) -> "PCMAndError": pass

    @abstractmethod
    def __rsub__(self, other: _Floatlike) -> "PCMAndError": pass

    @abstractmethod
    def __mul__(self, other: _Floatlike) -> "PCMAndError": pass

    @abstractmethod
    def __rmul__(self, other: _Floatlike) -> "PCMAndError": pass

    @abstractmethod
    def __truediv__(self, other: _Floatlike) -> "PCMAndError": pass

    @abstractmethod
    def __rtruediv__(self, other: _Floatlike) -> "PCMAndError": pass

    @abstractmethod
    def __eq__(self, other: _Floatlike) -> bool: pass

    @abstractmethod
    def __ne__(self, other: _Floatlike) -> bool: pass

    @abstractmethod
    def __hash__(self) -> int: pass

    @abstractmethod
    def __bool__(self) -> bool: pass

    @abstractmethod
    def __str__(self) -> str: pass

    @abstractmethod
    def __repr__(self) -> str: pass


def _pcm_err(res: KResult) -> PCMAndError:
    """Return the reactivity and error as a UFloat.

    """
    return ufloat(res.reactivity, res.reactivity_error)


def pcm_err(res: OracleResult) -> PCMAndError:
    """Return the reactivity and error as a UFloat.

    """
    return _pcm_err(res[KQuery()][0])


def diff(res1: OracleResult, res2: OracleResult) -> PCMAndError:
    """Calculate the reactivity difference in two results.

    """
    r1, r2 = map(pcm_err, (res1, res2))
    return r1 - r2


def split_name(name: PurePath
               ) -> Tuple[Optional[str], str,
                          Optional[float], Optional[float], Optional[float]]:
    """split name of component to it's site, the path from the site up to its parent and the position
    in space of the component"""
    name = str(name)
    if match := re.search('/Piece:[(]([^()/:]+)', name):
        x, y, z = map(float, match.groups()[0].split(', '))
    else:
        x = y = z = None
    if ':' in name:
        site = name.split('/')[0]
        label = '/'.join(name.split('/')[1:-1])
    else:
        site = None
        label = name
    return site, label, x, y, z


def invert(y_target: float, x: Sequence[float], y: Sequence[float]) -> float:
    """
    This function "inverts" a function described by the graph of the (x, y)
    pairs by assuming it is linear within successive x values. More specifically
    it approximates the x value of the function for which the function's
    value is y_target. It is assumed that x and y are ascending monotonically
    and are indexable.
    >>> invert(1.7, [0., 1.], [1., 2.])
    0.7
    """
    if not min(y) <= y_target <= max(y):
        raise ValueError(f'The target y value is not within the range of '
                         f'the input graph.')
    if len(x) != len(y):
        raise ValueError(f'The input x and y sequences are not of the same '
                         f'length, and thus do not represent a '
                         f'function\'s graph.')
    return interp(y_target, xp=y, fp=x).item()


significant_digit: Callable[[float], int] \
    = compose(int, negative, floor, log10)


def sround(value: uncertainties.UFloat) -> uncertainties.UFloat:
    """
    Round a UFloats by its significant digit, as determined by its standard
    deviation.
    >>> vals = [ufloat(120.4124124, 0.0545646), ufloat(53.95828, 1.24805780)]
    >>> vals
    [120.4124124+/-0.0545646, 53.95828+/-1.2480578]
    >>> list(map(sround, vals))
    [120.41+/-0.05, 54.0+/-1.0]
    """
    mean, std = juxt(nominal_value, std_dev)(value)
    n = significant_digit(std)
    return ufloat(round(mean, ndigits=n),
                  std_dev=round(std, ndigits=n))


def decompose_ufloats(seq: Sequence[uncertainties.UFloat]) \
        -> tuple[list[float], list[float]]:
    """
    Decompose a sequence of UFloats to two sequences of equal length, one
    for the nominal values, and one for the standard deviations.
    >>> vals = [ufloat(120.4124124, 0.0545646), ufloat(53.95828, 1.24805780)]
    >>> vals
    [120.4124124+/-0.0545646, 53.95828+/-1.2480578]
    >>> decompose_ufloats(vals)
    ([120.4124124, 53.95828], [0.0545646, 1.2480578])
    """
    means = list(map(nominal_value, seq))
    stds = list(map(std_dev, seq))
    return means, stds


OracleFuncFull = Callable[[OperationalState, Query], OracleResult]
OracleFunc = Callable[[OperationalState], KResult]
delayed = partial(unpure_delayed, pure=True)
DelayedOracleFunc = OracleFuncFull
