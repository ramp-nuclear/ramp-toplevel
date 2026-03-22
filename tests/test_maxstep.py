"""Tests for time step estimator functions"""

from datetime import timedelta

import hypothesis.strategies as st
from corecompute.result import KResult
from hypothesis import given
from scipy.constants import day

from ramp.search.eoc import max_step_deterministic as max_det

operational_maximums = st.timedeltas(min_value=timedelta(days=20.0), max_value=timedelta(days=40.0))
rhos = st.floats(min_value=200.0, max_value=5000.0)
drhos = st.floats(min_value=1.0, max_value=100.0)
kresults = st.builds(KResult.from_reactivity, rhos, drhos)


@given(
    op_max=operational_maximums,
    kres=kresults,
)
def test_deterministic_is_less_equal_op_max_in_general(op_max, kres):
    step = max_det(
        op_max_timestep=op_max,
        rho_target=0.0,
        drhodt=-100.0 / day,
        kres=kres,
    )
    assert step <= op_max


@given(kres=kresults)
def test_deterministic_returns_exact_time_to_target_when_small(kres: KResult):
    drdt = -100.0 / day
    expected = timedelta(seconds=(0.0 - kres.reactivity) / drdt)
    op_max = expected + timedelta(days=30)
    step = max_det(
        op_max_timestep=op_max,
        rho_target=0.0,
        drhodt=drdt,
        kres=kres,
    )
    assert abs(step.total_seconds() - expected.total_seconds()) < 1.0


@given(kres=kresults)
def test_deterministic_subtracts_minimal_step_when_just_over_op_max(kres: KResult):
    """When time-to-target slightly exceeds op_max, it subtracts minimal_timestep."""
    drdt = -100.0 / day
    exact = timedelta(seconds=(0.0 - kres.reactivity) / drdt)
    minimal = timedelta(days=1.0)
    # Set op_max so exact is just over it but within minimal_timestep
    op_max = exact - timedelta(hours=6)
    if op_max <= timedelta(0):
        return
    step = max_det(
        op_max_timestep=op_max,
        rho_target=0.0,
        drhodt=drdt,
        kres=kres,
        minimal_timestep=minimal,
    )
    if exact > op_max and exact - minimal < op_max:
        assert abs(step.total_seconds() - (exact - minimal).total_seconds()) < 1.0
    else:
        assert step <= op_max
