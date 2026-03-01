"""
This module help inspect an object pickle-dump by using a modified Pickler.
"""
import io
import pickle
from functools import partial
from typing import Iterable, List, TypeVar

from partd import Encode


class _Pickler(pickle._Pickler):
    def __init__(self, file, protocol=None, *, fix_imports=True,
                 buffer_callback=None):
        super().__init__(file, protocol=protocol)
        self.objects = {}

    def save_global(self, obj, name=None):
        super().save_global(obj, name)
        if name is None:
            name = getattr(obj, '__qualname__', None)
        if name is None:
            name = obj.__name__
        module_name = pickle.whichmodule(obj, name)
        key = f'{module_name}.{name}'
        self.objects.setdefault(key)


def find_objects(obj) -> list:
    """return names of objects in object pickle-dump"""
    f = io.BytesIO()
    pickler = _Pickler(f)
    pickler.dump(obj)
    return list(pickler.objects.keys())


T = TypeVar('T')


def _concat(lists: Iterable[List[T]]) -> T:
    try:
        y, = sum(lists, [])
    except ValueError as e:
        raise KeyError("Some of these keys were not found!") from e
    return y


Pickle = partial(Encode,
                 partial(pickle.dumps, protocol=pickle.HIGHEST_PROTOCOL),
                 lambda x: [pickle.loads(x)],
                 _concat)


class PickleWithKey(Encode):
    """Pickle and enter the key in the values.

    """
    def __init__(self, partd=None):
        super().__init__(partial(pickle.dumps, protocol=pickle.HIGHEST_PROTOCOL),
                         lambda x: [pickle.loads(x)],
                         _concat,
                         partd=partd)

    def append(self, data: dict, **kwargs):
        """Add a bunch of data in there.

        Parameters
        ----------
        data - Dictionary of data to add in.

        """
        super().append({k: (k, v) for k, v in data.items()}, **kwargs)

    def _iset(self, key, value, **kwargs):
        super()._iset(key, (key, value), **kwargs)

    iset = _iset

    def _get(self, keys, **kwargs):
        results = super()._get(keys, **kwargs)
        try:
            vkeys, values = map(list, zip(*results))
        except ValueError:  # It is for some reason acceptable to .get([]).
            return results  # In this case this edge case we will return []
        if not keys == vkeys:
            missing = set(vkeys).symmetric_difference(set(keys))
            raise KeyError(f"Some keys are missing: {missing}")
        return values
