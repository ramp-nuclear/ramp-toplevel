from collections import Counter
from copy import deepcopy
from functools import partial
from pathlib import PurePath

import dask
import numpy as np
import pytest
from coremaker.transform import Transform
from coreoperator import OperationalState
from coreoperator.example import example_state
from coreoperator.mobilization import LoadChain, CyclicShuffle, LoadSite, Scheme, Remove
from dask.delayed import Delayed
from more_itertools import first
from ramp.state_analysis.control_rod_worth import s_curve, moveable_reactivity_worth
from ramp.state_analysis.rod_worths import rods_extraction_worth
from ramp.state_analysis.shuffle_worths import decompose_action, \
    stepwise_shuffle_reactivity
from corecompute.query import KQuery
from corecompute.result import KResult
from toolz import identity
from uncertainties.core import AffineScalarFunc, nominal_value


def _to_oracle_result(reactivity, reactivity_error):
    reactivity = reactivity if isinstance(reactivity, (float, int)) else reactivity.item()
    reactivity_error = reactivity_error if isinstance(reactivity_error, float) else reactivity_error.item()
    return {KQuery(): [KResult.from_reactivity(reactivity, reactivity_error)]}


def test_get_rods_extraction_worth():
    rod_worth = 100.

    def _calculator(state, queries=None, prefix=''):
        aluminum_block_node = example_state.core[PurePath('A1') / PurePath('coolant') / PurePath('aluminum_block')]
        num_of_aluminum_blocks = sum(node == aluminum_block_node
                                     for _, node in state.core.nodes)
        nominal_num_of_aluminum_blocks = np.prod(example_state.core.grid.lattice.shape)
        reactivity = rod_worth * \
                     (num_of_aluminum_blocks - nominal_num_of_aluminum_blocks)
        reactivity_error = 0.03 * abs(reactivity)
        return _to_oracle_result(reactivity, reactivity_error)

    sites = [key for key in example_state.core.grid.contents.keys()][:4]
    delayed_rods_worths = rods_extraction_worth(example_state, sites,
                                                calculator=_calculator)
    rods_worth,  = dask.compute(delayed_rods_worths)
    assert list(rods_worth.keys()) == sites
    assert all([np.isclose(rods_worth[site].nominal_value, -rod_worth)
                for site in sites])


def _get_control_height(state, alias):
    heights = {path: state.core.transform_of(path).translation[2]
               for path in state.core.aliases[alias][1]}
    height = first(heights.values())
    # Checking that all control rods are at the same height.
    assert all(h == height for h in heights.values())
    return height


def _control_heights_based_calculator(state, nominal_state, worth_per_cm,
                                      queries=None, prefix=''):
    nominal_height = _get_control_height(nominal_state, 'All')
    height = _get_control_height(state, 'All')
    reactivity = worth_per_cm * (height - nominal_height)
    reactivity_error = 0.03 * abs(reactivity)
    return _to_oracle_result(reactivity, reactivity_error)


def test_scurve():

    worth_per_cm = 200.

    def _calculator(state, queries=None, prefix=''):
        return _control_heights_based_calculator(state, example_state,
                                                 worth_per_cm,
                                                 queries=queries, prefix=prefix)

    heights = np.linspace(-30.0, 30.0, 20)
    delayed_scurve_results = s_curve(example_state, 'All', heights,
                                     calculator=_calculator)
    scurve_results, = dask.compute(delayed_scurve_results)
    assert np.allclose(heights, np.fromiter(scurve_results.keys(),
                                            dtype=float))
    reactivities = np.fromiter((map(nominal_value, scurve_results.values())),
                               dtype=float)
    ratio = np.diff(reactivities) / np.diff(heights)
    assert np.allclose(ratio, worth_per_cm, rtol=1e-4, atol=0.)


def test_movable_reactivity_worth():

    worth_per_cm = 200.

    def _calculator(state, queries=None, prefix=''):
        return _control_heights_based_calculator(state, example_state,
                                                 worth_per_cm,
                                                 queries=queries, prefix=prefix)

    inserted = -30.
    extracted = 30.
    delayed_moveable_worth = moveable_reactivity_worth(example_state, 'All',
                                                       inserted, extracted,
                                                       calculator=_calculator)
    assert isinstance(delayed_moveable_worth, Delayed)
    moveable_worth, = dask.compute(delayed_moveable_worth)
    assert isinstance(moveable_worth, AffineScalarFunc)
    assert np.isclose(moveable_worth.nominal_value,
                      worth_per_cm * (extracted - inserted), rtol=1e-6, atol=0.)


@pytest.fixture()
def factory():
    factory = partial(identity, deepcopy(example_state.core.grid['A1']))
    factory.__name__ = 'A1'
    return factory


def test_decompose_action(factory):
    sites = list(example_state.core.grid.sites())
    transforms = [Transform((0., 0., z))
                  for z in np.linspace(0., 100., len(sites))]
    load_chain = LoadChain(factory=factory,
                           sites=list(zip(sites, transforms)))
    decomposed_action = decompose_action(load_chain, example_state)
    assert len(decomposed_action) == 2 * len(sites)
    assert all(isinstance(action, Remove) | isinstance(action, LoadSite)
               for action in decomposed_action)
    sites_in_decomposed_action = [first(action.sites)
                                  if isinstance(action, Remove)
                                  else first(action.sites)[0]
                                  for action in decomposed_action]
    assert all(site in sites for site in sites_in_decomposed_action), [site for site in sites_in_decomposed_action if site not in sites]
    counter = Counter(sites_in_decomposed_action)
    assert all(counter[site] == 2 for site in sites)
    assert all(first(action.sites)[1] == t
               for action, t in zip(filter(lambda x: isinstance(x, LoadSite),
                                           decomposed_action), reversed(transforms)))
    composed_scheme = Scheme((load_chain, ))
    decomposed_scheme = Scheme(tuple(decomposed_action))
    composed_state = example_state.new_after_scheme(composed_scheme)
    decomposed_state = example_state.new_after_scheme(decomposed_scheme)
    for site in sites:
        assert composed_state.core.grid[site] == decomposed_state.core.grid[site]


def test_stepwise_shuffle_reactivity(factory):
    sites = list(example_state.core.grid.sites())[:10]
    load_chain = LoadChain(factory=factory,
                           sites=sites)
    scheme = Scheme((load_chain,))

    def _calculator(state: OperationalState, queries=None, prefix=''):
        if not isinstance(state.history.steps[-1], Scheme):
            reactivity = 0.
        else:
            reactivity = len(state.history.steps[-1].actions)
        reactivity_error = 0.03 * reactivity
        return _to_oracle_result(reactivity, reactivity_error)
    reactivities, = dask.compute(stepwise_shuffle_reactivity(example_state,
                                                             scheme,
                                                             calculator=_calculator))
    for scheme, reactivity in reactivities.items():
        assert np.isclose(len(scheme.actions), nominal_value(reactivity))
    assert len(reactivities) == 2 * len(sites) + 1

