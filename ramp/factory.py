"""A tool used to create operational states from nothing."""

from typing import Generic, TypeVar

from coreoperator.operational_state import OperationalState

Seed = TypeVar("Seed")


class Factory(Generic[Seed]):
    """A protocol for how an operational state factory should behave."""

    def __call__(self, seed: Seed) -> OperationalState:
        pass
