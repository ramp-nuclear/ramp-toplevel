from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from random import Random
from typing import Sequence, Iterable, Optional

from batman import BurnResult
from batman.units import PCM
from coreoperator.history import History, OperationalPeriod, StateParams
from scipy.constants import day

from corecompute.oracle import OracleResult
from corecompute.query import KQuery
from corecompute.result import KResult


class FakeState:
    def __init__(self, control_worths: dict[str, float],
                 aliases: dict[str, Sequence[str]]):
        self.control_worths = control_worths
        self.aliases = aliases
        self.heights = {alias: 0. for alias in control_worths.keys()}
        self.time = timedelta(0.)
        self.drhodt = -100. / day
        self.rho_0_in = -3000.
        self.out_h = 64.
        self.history = History([OperationalPeriod(StateParams(power=25., **aliases), time=timedelta(1e-3))])

    @property
    def controls(self) -> Iterable[tuple[float, float]]:
        for alias, height in self.heights.items():
            yield height, self.control_worths[alias]

    @classmethod
    def new_control_height(cls, state: "FakeState",
                           alias: str, height: float) -> "FakeState":
        new_state = deepcopy(state)
        for key in new_state.aliases[alias]:
            new_state.heights[key] = height
        return new_state

    @classmethod
    def new_time(cls, state: "FakeState", timestep: timedelta):
        new_state = deepcopy(state)
        new_state.time += timestep
        pars = state.history.current_params.copy()
        new_state.history.timestep(time=timestep, params=pars)
        return new_state


def _except_v(d: list, v):
    e = d.copy()
    e.remove(v)
    return e


class FakeClient:
    @contextmanager
    def as_current(self):
        yield


class FakeOracle:
    def __init__(self, error: PCM = 75.):
        self.client = FakeClient()
        self.error = error

    def direct(self, state: FakeState, *queries, **_) -> OracleResult:
        rho = (state.rho_0_in
               + sum(w * h / state.out_h for h, w in state.controls)
               + state.time.total_seconds() * state.drhodt)
        kres = KResult.from_reactivity(rho, self.error)
        return {query: [kres] if query == KQuery() else None
                for query in queries}

    __call__ = direct


class FakeOracleNoisy(FakeOracle):

    def __init__(self, error: PCM, seed: int = 42, limit: float = 2.):
        super().__init__(error=error)
        self._rand = Random(seed)
        self.limit = limit

    def _random_but_limited(self, mu: PCM, std: PCM,
                            limit: Optional[float] = None) -> PCM:
        limit = limit or self.limit
        noise = mu - 5*limit*std
        while not (mu - limit*std <= noise <= mu + limit*std):
            noise = self._rand.normalvariate(mu, std)
        return noise

    def direct(self, state: FakeState, *queries, **__) -> OracleResult:
        rho = (state.rho_0_in
               + sum(w * h / state.out_h for h, w in state.controls)
               + state.time.total_seconds() * state.drhodt)
        noise = self._random_but_limited(rho, self.error)
        kres = KResult.from_reactivity(noise, self.error)
        return {query: [kres] if query == KQuery() else None
                for query in queries}

    __call__ = direct


def _new_k(k0: float, drhodt: float, time: float) -> tuple[float, float]:
    kres = KResult(k0, 0.)
    rho_new = kres.reactivity + drhodt * time
    knew = KResult.from_reactivity(rho_new, 0.).k
    dkdt = (1e-5 * knew**2) * drhodt
    return knew, dkdt


class FakeBatman:
    def __call__(self, state: FakeState, time: timedelta, k0: float, **_):
        knew, dkdt = _new_k(k0, state.drhodt, time.total_seconds())
        batinfo = BurnResult(1, time, 1., knew, dkdt)
        return FakeState.new_time(state, time), batinfo

    def burn_pcm(self, state: FakeState, maximal_timestep: timedelta,
                 k: float, drho: PCM, **_):
        trho = timedelta(seconds=-drho / state.drhodt)
        if trho < timedelta(0.):
            raise ValueError("Cannot step backwards")
        t = min(trho, maximal_timestep)
        knew, dkdt = _new_k(k, state.drhodt, t.total_seconds())
        batinfo = BurnResult(1, t, 1., knew, dkdt)
        return FakeState.new_time(state, t), batinfo

    def queries(self, *_):
        return tuple()
