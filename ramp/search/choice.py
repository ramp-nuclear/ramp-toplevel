"""Tools for running a bunch of function calls with scores to find the best one.

"""
from operator import itemgetter
from typing import Callable, Sequence, Any, Union, Tuple
from functools import partial
import logging

from dask import delayed

from coreoperator.operational_state import OperationalState
from ramp.factory import Factory, Seed

ScoreFunc = Union[Callable[[OperationalState], Any], partial]
logger = logging.getLogger('distributed.ramp.choicer')


def best_choice(factory: Factory[Seed],
                scoring_function: ScoreFunc,
                guesses: Sequence[Seed]) -> Tuple[Seed, OperationalState, Any]:
    """Pick the best seed in terms of a scoring function.

    Parameters
    ----------
    factory - Tool to turn a seed into a calculatable object.
    scoring_function - Function to use when scoring.
    guesses - Seeds to pick from

    Returns
    -------
    3-tuple of the best seed, resulting best state and additional data from the
    scoring function.

    """
    scoring = partial(_zip_score, factory=factory, scoring=scoring_function)
    scores = [delayed(scoring, pure=True)(guess) for guess in guesses]
    seed, *data = delayed(max)(scores, key=itemgetter(0))[1:].compute()
    return seed, factory(seed), *data


def _zip_score(seed: Seed, factory: Factory[Seed], scoring: ScoreFunc) -> \
        Tuple[float, Seed, Any]:
    state = factory(seed)
    score, *data = scoring(state)
    return score, seed, *data
