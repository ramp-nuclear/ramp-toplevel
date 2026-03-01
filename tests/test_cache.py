"""Tests for shelving.

"""
import logging
from collections import Counter
from functools import partial
from pathlib import Path
from string import ascii_lowercase
from typing import Dict

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from partd.core import Interface

from ramp.utils.cached import file_cache, FileShortKey


test_dir = Path(__file__).parent / 'test_data'
logger = logging.getLogger('distributed.test_shelve')

filedbs = st.builds(FileShortKey)
keys = st.text([letter for letter in ascii_lowercase]
               + [n for n in '0123456789'],
               min_size=100, max_size=6000)
pickfiles = st.builds(partial(file_cache, None))


@settings(max_examples=200)
@given(db=filedbs, key=keys)
def test_filename_never_large(db: FileShortKey, key: str):
    with db:
        pieces = db.filename(key).split('/')
        assert all(len(piece) <= 64 for piece in pieces)


keylists = st.lists(keys, max_size=1000, unique=True)


@settings(max_examples=100)
@given(db=filedbs, _keys=keylists)
def test_filenames_do_not_clash(db: FileShortKey, _keys: Dict[str, int]):
    fnames = (db.filename(key) for key in _keys)
    fcounter = Counter(fnames)
    assert not fcounter or fcounter.most_common(1)[0][1] <= 1


@settings(max_examples=100)
@given(db=pickfiles, key=keys, value=st.integers())
def test_file_holds_single_item_correctly(db: Interface, key: str, value: int):
    with db:
        db.iset(key, value)
        assert db.iget(key) == value


@given(db=pickfiles, key=keys, value=st.integers(min_value=0))
def test_file_is_updated(db: Interface, key: str, value: int):
    db.iset(key, value-1)
    assert db.iget(key) == value-1
    db.iset(key, value)
    assert db.iget(key) == value


@pytest.fixture
def _shelve_db() -> Interface:
    return file_cache(test_dir / 'oracle_shelvy')
