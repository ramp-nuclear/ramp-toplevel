"""Module for common way to report k eigenvalue results.

"""
from statistics import NormalDist

PCM = float

__all__ = ['KResult', 'PCM']


class KResult:
    """Result object for k-eigenvalue results

    TODO: Replace this implementation with uncertainties' ufloat

    """

    def __init__(self, k: float, dk: float):
        r"""

        Parameters
        ----------
        k - k-eigenvalue
        dk - Error in k-eigenvalue. This usually means 1-\sigma standard
             deviation, but in other cases it can also mean just the
             convergence error, which is bounded rather than distributed
             normally.
        """

        self.k = k
        self.dk = dk

    def __str__(self):
        return f'KResult({self.k:.2f} ± {self.dk:.2f})'

    __repr__ = __str__

    @property
    def rho_dist(self) -> NormalDist:
        """The distribution of the reactivity according to the data.

        """
        return NormalDist(mu=self.rho, sigma=self.drho)

    @classmethod
    def from_reactivity(cls, rho: PCM, drho: PCM) -> "KResult":
        r"""Create a k-result from reactivity values.

        Parameters
        ----------
        rho - Reactivity of the core, in PCM
        drho - Error in terms of reactivity, in PCM. Currently used only for
               1-\sigma standard deviation. Other errors need to be included
               in the future.

        """

        k = 1. / (1. - (rho / XXX))
        return cls(k=k, dk=(k ** 2) * (drho / XXX))

    # noinspection PyMissingOrEmptyDocstring
    @property
    def reactivity(self) -> PCM: return 1e5 * (1. - (1. / self.k))

    rho = reactivity

    # noinspection PyMissingOrEmptyDocstring
    @property
    def reactivity_error(self) -> PCM: return 1e5 * (self.dk / (self.k ** 2))

    drho = reactivity_error
