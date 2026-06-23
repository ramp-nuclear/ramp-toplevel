"""Tests of reaching EOC or performing a transport-burnup pair."""

from datetime import timedelta

import hypothesis.strategies as st
from hypothesis import given

from ramp.regime.regime import Regime
from ramp.search.eoc import find_eoc

from .mocks import FakeBatman, FakeOracle, FakeOracleNoisy, fake_state_factory, shut_ramp_up


@shut_ramp_up
def test_reach_eoc_with_fake_exact():
    boc = fake_state_factory()
    oracle = FakeOracle(error=75.0)
    batman = FakeBatman()
    faked_regime = Regime(
        maximal_timestep=timedelta(days=20.0),
        initial_steps=[timedelta(days=3.0)],
        oracle=oracle,
        batman=batman,
    )
    boc = boc.new_control_height("All", boc.out_h)
    bo3, eoc = find_eoc(state=boc, regime=faked_regime, rho=0.0)
    kwild = faked_regime.get_kwild(eoc)
    assert abs(kwild.reactivity) < 100.0 + kwild.reactivity_error


@shut_ramp_up
@given(st.integers())
def test_reach_eoc_with_fake_noisy(seed: int):
    fake_state = fake_state_factory()
    oracle = FakeOracleNoisy(error=25.0, seed=seed, limit=1.5)
    batman = FakeBatman()
    faked_regime = Regime(
        maximal_timestep=timedelta(days=20.0),
        initial_steps=[timedelta(days=3.0)],
        oracle=oracle,
        batman=batman,
    )
    boc = fake_state.new_control_height("All", fake_state.out_h)
    bo3, eoc = find_eoc(state=boc, regime=faked_regime, rho=0.0)
    kwild = faked_regime.get_kwild(eoc)
    assert abs(kwild.reactivity) < 100.0 + kwild.reactivity_error
