MW = float


class HeatingRateQuery:
    """Query about the heating rate per neutron source. At units of J per neutron source.
    There are a lot of different methods to estimate the power, depending on what reactions are accounted for.
    The method will be specified at the creation of the query, and each oracle might support different methods.
    For example the openmc oracle supports the use of the: 'score heating','score heating-local', 'score kappa-fission',
    'score fission-q-prompt' and 'score fission-q-recoverable' which use the corresponding openmc scores

    """

    def __init__(self, method: str, **kwargs):
        """
        method: str
         The name of the method to be used
        kwargs: Dict
         keyword arguments used by the specified method
        """
        self.method = method
        self.__dict__.update(kwargs)

    def __hash__(self) -> int:
        return hash(None)

    def __eq__(self, other: "PowerQuery") -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.method==other.method

    def __repr__(self):
        return f"HeatingRateQuery Using method {self.method}"


