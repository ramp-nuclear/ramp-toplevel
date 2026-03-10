"""Tests for time step estimator functions"""

from datetime import timedelta

from hypothesis import given, settings
import hypothesis.strategies as st
from scipy.constants import day

from ramp.search.eoc import max_safe_step_at_risk as max_safe
from corecompute.result import KResult


operational_maximums = st.timedeltas(
    min_value=timedelta(days=20.0), max_value=timedelta(days=40.0)
)
rhos = st.floats(min_value=200.0, max_value=5000.0)
drhos = st.floats(min_value=1.0, max_value=100.0)
kresults = st.builds(KResult.from_reactivity, rhos, drhos)


@settings(max_examples=500)
@given(
    op_max=operational_maximums,
    kres=st.builds(KResult.from_reactivity, st.floats(min_value=3e3, max_value=1e4), drhos)
)
def test_max_safe_is_op_max_if_takes_super_long_to_reach(op_max, kres):
    step = max_safe(
        op_max_timestep=op_max,
        rho_target=0.0,
        drhodt=-10.0 / day,
        kres=kres,
    )
    assert step == op_max


@settings(max_examples=500)
@given(
    op_max=operational_maximums,
    kres=kresults,
)
def test_max_safe_is_less_equal_op_max_in_general(op_max, kres):
    step = max_safe(
        op_max_timestep=op_max,
        rho_target=0.0,
        drhodt=-100.0 / day,
        kres=kres,
    )
    assert step <= op_max


@settings(max_examples=500)
@given(kres=kresults)
def test_max_safe_less_than_op_max_if_wanted_close_to_op_max(kres: KResult):
    rho0 = kres.reactivity
    drdt = -100.0
    needed = timedelta(days=- rho0/drdt)
    op_max = needed - timedelta(hours=4)
    step = max_safe(
        op_max_timestep=op_max,
        rho_target=0.0,
        drhodt=drdt,
        kres=kres,
    )
    assert step < op_max


@settings(max_examples=500)
@given(kres=kresults)
def test_max_safe_smaller_if_sigma_is_bigger(kres: KResult):
    rho0 = kres.reactivity
    drdt = -100.0
    op_max = timedelta(weeks=5000)
    kres2 = KResult.from_reactivity(rho0, kres.reactivity_error * 2)
    step1 = max_safe(
        op_max_timestep=op_max,
        rho_target=0.0,
        drhodt=drdt,
        kres=kres,
    )
    step2 = max_safe(
        op_max_timestep=op_max,
        rho_target=0.0,
        drhodt=drdt,
        kres=kres2,
    )
    assert step2 < step1
