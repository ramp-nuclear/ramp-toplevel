"""The Oracle protocol, used to define how Transport related queries are
answered.

"""
from functools import partial
from typing import Protocol, Dict, Sequence, Callable, Union

from distributed import Client, Future

from coreoperator.operational_state import OperationalState
from ramp.transport.query import Query
from ramp.transport import Result


OracleResult = Dict[Query, Sequence[Result]]


class Oracle(Protocol):
    """
    An Oracle is used to answer queries about operational states.
    """

    direct: Union[partial, Callable]

    def __call__(self, state: OperationalState,
                 *queries: Query) -> OracleResult:
        pass
