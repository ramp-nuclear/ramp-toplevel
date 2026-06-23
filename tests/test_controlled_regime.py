from datetime import timedelta

from ramp.regime import ControlledRegime

from .mocks import FakeBatman, FakeOracle, fake_state_factory, shut_ramp_up


def _regime():
    return ControlledRegime(
        maximal_timestep=timedelta(days=30.0),
        initial_steps=[],
        oracle=FakeOracle(error=10.0),
        guesses=[0.0, 10.0, 20.0],
        outside=20.0,
        control_alias="All",
        batman=FakeBatman(),
    )


@shut_ramp_up
def test_controlled_regime_chooses_new_height_for_fake_state():
    regime = _regime()
    con = regime.get_controlled_state(fake_state_factory())
    assert con.heights != fake_state_factory().heights


@shut_ramp_up
def test_controlled_regime_cache_not_empty_after_controlled():
    regime = _regime()
    regime.get_controlled_state(fake_state_factory())
    assert len(regime.result_per_height) == 1


@shut_ramp_up
def test_controlled_regime_step_empties_cache_of_previous_state():
    regime = _regime()
    s = regime.get_controlled_state(fake_state_factory())
    d1 = {k: v for k, v in regime.result_per_height.items()}
    regime.burnstep(s, timedelta(3.0))
    assert d1 != regime.result_per_height
