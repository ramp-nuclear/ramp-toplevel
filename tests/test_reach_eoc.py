"""Tests of reaching EOC or performing a transport-burnup pair.

"""
import logging
from datetime import timedelta
from functools import wraps

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from ramp.regime.regime import Regime
from ramp.search.eoc import find_eoc
from .mocks import FakeState, _except_v, FakeOracle, FakeOracleNoisy, FakeBatman


slow = pytest.mark.slow
logger = logging.getLogger(__name__)


def _shut_ramp_up(f):
    @wraps(f)
    def _wrapper(*args, **kwargs):
        ramplogger = logging.getLogger('ramp')
        prevlevel = ramplogger.level
        ramplogger.level = logging.WARNING
        res = f(*args, **kwargs)
        ramplogger.level = prevlevel
        return res
    return _wrapper


def fake_state_factory() -> FakeState:
    """Create a fake OperationalState

    """
    blades = (f'Blade{i}' for i in range(6))
    worths = (3000., 3000., 3000., 3000., 2000., 2000.)
    controls = dict(zip(blades, worths))
    aliases = {f'R_{v}': _except_v(list(controls.keys()), v)
               for v in controls.keys()}
    aliases['All'] = list(controls.keys())
    return FakeState(controls, aliases)


fake_states = st.builds(fake_state_factory)


@_shut_ramp_up
@settings(max_examples=500)
@given(boc=fake_states)
def test_reach_eoc_with_fake_exact(boc: FakeState):
    oracle = FakeOracle(error=75.)
    batman = FakeBatman()
    # noinspection PyTypeChecker
    faked_regime = Regime(maximal_timestep=timedelta(days=20.),
                          initial_steps=[timedelta(days=3.)],
                          oracle=oracle, batman=batman)
    boc = boc.new_control_height(boc, 'All', boc.out_h)
    bo3,eoc = find_eoc(state=boc, regime=faked_regime, rho=0.)
    kwild = faked_regime.get_kwild(eoc)
    assert abs(kwild.reactivity) < 100. + kwild.reactivity_error


@_shut_ramp_up
@settings(max_examples=500)
@given(seed=st.integers(), fake_state=fake_states)
def test_reach_eoc_with_fake_noisy(fake_state: FakeState, seed: int):
    oracle = FakeOracleNoisy(error=75., seed=seed, limit=1.5)
    batman = FakeBatman()
    # noinspection PyTypeChecker,PyTypeChecker
    faked_regime = Regime(maximal_timestep=timedelta(days=20.),
                          initial_steps=[timedelta(days=3.)],
                          oracle=oracle, batman=batman)
    boc = fake_state.new_control_height(fake_state, 'All', fake_state.out_h)
    bo3,eoc = find_eoc(state=boc, regime=faked_regime, rho=0.)
    kwild = faked_regime.get_kwild(eoc)
    assert abs(kwild.reactivity) < 100. + kwild.reactivity_error
