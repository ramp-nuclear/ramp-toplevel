"""Tools for expanding documentation when inheriting stuff."""

from typing import Callable

__all__ = ["append_doc_of"]


def append_doc_of(superb: Callable):
    """Append the documentation of another function after the documentation for
    the decorated function. Basically a decorator factory.

    Parameters
    ----------
    superb - Function to take documentation from.

    """

    def _deco(func: Callable):
        doc = func.__doc__ or ""
        func.__doc__ = doc + superb.__doc__
        return func

    return _deco


def _moo():
    """I have the best documentation ever. Period."""
    return 3


def test_appending_for_simple_case():
    @append_doc_of(_moo)
    def _foo():
        """I have some documentation."""
        return 4

    assert _foo.__doc__.endswith(_moo.__doc__)


def test_appending_for_undocumented():
    @append_doc_of(_moo)
    def _bar():
        return 5

    assert _bar.__doc__ == _moo.__doc__
