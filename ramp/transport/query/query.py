from typing import Protocol


class Query(Protocol):
    """Type hint for queries. They must be hashable.

    """

    def __hash__(self) -> int:
        ...

    def __eq__(self, other) -> bool:
        ...

    def __repr__(self) -> str:
        ...


class KQuery:
    """Query about the system k eigenvalue and eigenvector

    """

    def __hash__(self) -> int:
        return hash(None)

    def __eq__(self, other: "KQuery") -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return True

    def __repr__(self) -> str:
        return "KQuery()"
