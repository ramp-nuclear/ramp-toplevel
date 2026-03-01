"""Tests that our custom partd-like Pickle objects are consistent with partd.

"""
import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from partd import Dict
from ramp.utils.pickle import Pickle, PickleWithKey

a = st.deferred(lambda: st.integers() | b | c)
b = st.deferred(lambda: st.lists(a | b | c))
c = st.deferred(lambda: st.dictionaries(st.integers(), a | b | c))
dct = st.builds(Dict)
regpickles = st.builds(Pickle, dct)
keypickles = st.builds(PickleWithKey, dct)
pickled = regpickles | keypickles


class _GoodDict(Dict):
    iset = Dict._iset


@settings(max_examples=800)
@given(value=a, p=pickled)
def test_pickle_dict_gets_same_value_that_sets(value, p):
    p.iset(0, value)
    assert p.iget(0) == value


@settings(max_examples=800)
@given(value=a, p=pickled)
def test_pickle_load_of_multiple_keys_is_multiple_results(value, p):
    p.append({i: value for i in range(100)})
    res = p.get(list(range(100)))
    assert res == [value for _ in range(100)]


@given(interface=pickled)
def test_grab_missing_key_raises_keyerror(interface):
    key = 'moo'
    with pytest.raises(KeyError):
        interface.iget(key)
    with pytest.raises(KeyError):
        interface.get([key])


@given(interface=pickled)
def test_grab_missing_some_keys_raises_keyerror(interface):
    initial = {'a': 'z', 'b': 'y', 'c': 'x'}
    interface.append(initial)
    keys = ['a', 'b', 'moo']
    with pytest.raises(KeyError):
        interface.get(keys)


@settings(max_examples=400)
@given(regp=regpickles, keyp=keypickles, key=st.integers(), value=a)
def test_pickle_with_key_same_as_without(regp, keyp, key, value):
    regp.iset(key, value)
    keyp.iset(key, value)
    assert regp.iget(key) == keyp.iget(key)


multikeys = st.lists(st.integers())
multivals = st.lists(a)


@given(regp=regpickles, keyp=keypickles, keys=multikeys, values=multivals)
def test_pickle_with_key_same_as_without_multiple(regp, keyp, keys, values):
    appendix = dict(zip(keys, values))
    regp.append(appendix)
    keyp.append(appendix)
    inkeys = list(appendix.keys())
    assert regp.get(inkeys) == keyp.get(inkeys)


gooddicts = st.builds(_GoodDict)
goodpickles = st.builds(PickleWithKey, gooddicts)


@given(interface=goodpickles, key=st.integers(), value=st.integers())
def test_pickle_key_saving_value_changes_value(interface, key, value):
    init_val = value - 1
    interface.iset(key, init_val)
    assert init_val == interface.iget(key)
    interface.iset(key, value)
    assert value == interface.iget(key)
    assert value != init_val
