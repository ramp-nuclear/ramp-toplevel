"""Oracle that looks things up in a shelve first.

"""

import logging
from functools import wraps
from inspect import signature
from typing import Optional, Union, Tuple, Sequence

from coreoperator.operational_state import OperationalState
from decorator import decorator
from distributed import Future
from partd.core import Interface

from ramp.oracle import OracleResult
from ramp.transport.query import Query
from ramp.utils.cached import get_key

logger = logging.getLogger('distributed.ramp.cached_oracle')


def _printable_key(state: OperationalState) -> str:
    """Get a printable version of the key in the cache

    Parameters
    ----------
    state - State to get the key for.

    """
    return str((state.history.timedelta, state.design_name))


class _CachedFuture:
    def __init__(self, future: Union[Future, OracleResult]):
        self._result = future

    def result(self, *args, **kwargs) -> OracleResult:
        """Returns either cached results or blocks until the future returns,
        depending on if this contains an actual future or not.

        Parameters
        ----------
        args, kwargs - Stuff to send to Future.result()

        Returns
        -------
        Definitively returns the actual queries results.

        """
        if not isinstance(self._result, Future):
            return self._result
        else:
            return self._result.result(*args, **kwargs)

    def __getattr__(self, item):
        if isinstance(self._result, Future):
            return self._result.__getattribute__(item)
        else:
            raise AttributeError(f"{self} has no attribute: {item}")


@decorator
def _cache_future(f, *args, **kwargs) -> _CachedFuture:
    """Decorator that takes an Oracle future function, and wraps that result
    in a _CachedFuture. This allows a lookup to be hidden as a future.

    Parameters
    ----------
    f - Function to decorate. Must return either a Future or an OracleResult.

    """
    return _CachedFuture(f(*args, **kwargs))


def get_cached_results(cache: Interface, state: OperationalState,
                       *queries: Query) -> OracleResult:
    """Get some cached results from the cache

    Parameters
    ----------
    cache - Factory used to generate the persistent cache interface.
    state - State to look up results for.
    queries - Queries that should appear for this to be treated as ok.

    Raises
    ------
    KeyError if results are not found for the state or if some query data
    is missing.

    """
    try:
        results: OracleResult = cache.iget(get_key(state))
    except Exception as e:
        logger.debug(f"Could not find {get_key(state)} in the cache "
                     "service")
        raise KeyError(f"Could not find {get_key(state)} "
                       "due to some other error") from e
    missing = set(queries) - set(results.keys())
    if missing:
        logger.debug(f"Some queries were missing: %r", missing)
        raise KeyError(f"Some queries are missing: {missing}")
    return results


def _shelf_state_queries(
        f, cache: Optional[Interface],
        args, state: Optional[str] = None
        ) -> Tuple[Interface, OperationalState, Sequence[Query]]:
    first_arg_is_self = next(iter(signature(f).parameters.keys())) == 'self'
    cache = args[0].cache if not cache and first_arg_is_self else cache
    skip = 1 if first_arg_is_self else 0
    queries = args[skip:] if state else args[skip + 1:]
    state = state or args[skip]
    return cache, state, queries


def _try_cache_first(cache: Optional[Interface] = None):
    def _deco(f):
        @wraps(f)
        def _f(*args, **kwargs) -> OracleResult:
            _cache, state, queries = _shelf_state_queries(f, cache, args)
            try:
                return get_cached_results(_cache, state, *queries)
            except KeyError:
                return f(*args, **kwargs)

        return _f

    return _deco


def _write_results_after(cache: Optional[Interface] = None):
    def _deco(f):
        @wraps(f)
        def _f(*args, **kwargs) -> OracleResult:
            raw = f(*args, **kwargs)
            _cache, state, _ = _shelf_state_queries(f, cache, args)
            results = {key: list(value) for key, value in
                       raw.items()}
            _cache.iset(get_key(state), results)
            return results

        return _f

    return _deco

