"""Tests for time step estimator functions

"""
from datetime import timedelta

from hypothesis import given, settings
import hypothesis.strategies as st

from ramp.search.eoc import max_safe_step_at_risk as max_safe
from ramp.transport.result import KResult


operational_maximums = st.timedeltas(min_value=timedelta(days=20.),
                                     max_value=timedelta(days=40.))
rhos = st.floats(min_value=200., max_value=5000.)
drhos = st.floats(min_value=1., max_value=100.)
kresults = st.builds(KResult.from_reactivity, rhos, drhos)


@settings(max_examples=500)
@given(op_max=operational_maximums,
       wanted=st.timedeltas(min_value=timedelta(0),
                            max_value=timedelta(days=20.)),
       kres=kresults)
def test_max_safe_is_op_max_for_small_wanted(op_max, wanted, kres):
    step = max_safe(op_max_timestep=op_max,
                    wanted_timestep=wanted,
                    rho_target=0.,
                    drhodt=-100.,
                    kres=kres)
    assert step == op_max


@settings(max_examples=500)
@given(op_max=operational_maximums,
       wanted=st.timedeltas(min_value=timedelta(0),
                            max_value=timedelta(days=50.)),
       kres=kresults)
def test_max_safe_is_less_equal_op_max_in_general(op_max, wanted, kres):
    step = max_safe(op_max_timestep=op_max,
                    wanted_timestep=wanted,
                    rho_target=0.,
                    drhodt=-100.,
                    kres=kres)
    assert step <= op_max


@settings(max_examples=500)
@given(kres=kresults)
def test_max_safe_less_than_op_max_if_wanted_close_to_op_max(kres: KResult):
    rho0 = kres.reactivity
    drdt = -100.
    wanted = timedelta(seconds=-rho0 / drdt)
    op_max = wanted - timedelta(hours=4)
    step = max_safe(op_max_timestep=op_max,
                    wanted_timestep=wanted,
                    rho_target=0.,
                    drhodt=drdt,
                    kres=kres)
    assert step < op_max


@settings(max_examples=500)
@given(kres=kresults)
def test_max_safe_smaller_if_sigma_is_bigger(kres: KResult):
    rho0 = kres.reactivity
    drdt = -100.
    wanted = timedelta(seconds=-rho0 / drdt)
    op_max = wanted - timedelta(hours=4)
    kres2 = KResult.from_reactivity(rho0, kres.reactivity_error*2)
    step1 = max_safe(op_max_timestep=op_max,
                     wanted_timestep=wanted,
                     rho_target=0.,
                     drhodt=drdt,
                     kres=kres)
    step2 = max_safe(op_max_timestep=op_max,
                     wanted_timestep=wanted,
                     rho_target=0.,
                     drhodt=drdt,
                     kres=kres2)
    assert step2 < step1
