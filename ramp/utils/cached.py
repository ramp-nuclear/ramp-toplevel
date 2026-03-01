"""Tools for caching very costly functions.

"""
import codecs
import pickle
from functools import partial, wraps
from pathlib import Path
from typing import Callable, Union
from hashlib import sha256

from numpy import base_repr
from partd import File, Client, Server
from partd.core import Interface
from partd.zmq import NotALock
from distributed.utils import get_ip_interface

from coreoperator.operational_state import OperationalState

from .pickle import PickleWithKey as Pickle


MAXLEN = 60


def _grab_key_pieces(state: OperationalState):
    return state.design_name, state.history


def encode_key(obj) -> str:
    """Encode a key object as a string. This may be a large string.

    """
    return codecs.encode(pickle.dumps(obj), "base64").decode()


def get_key(state: OperationalState) -> str:
    """Encode a state to a bite sized string for key in a cache.

    """
    return encode_key(_grab_key_pieces(state))


def decode_key(key: str):
    """Decode what a string encoded key means.

    Parameters
    ----------
    key - Gibberish key to decode

    Returns
    -------

    Some object that was pickled, encoded and decoded to make this key.

    """
    return pickle.loads(codecs.decode(key.encode(), "base64"))


def _shorten_second_arg(f):
    @wraps(f)
    def _wrapped(self, keys, *args, **kwargs):
        return (f(self, type(keys)(map(self._shortkey, keys)),
                  *args, **kwargs)
                if isinstance(keys, (list, tuple, set))
                else f(self, self._shortkey(keys), *args, **kwargs))
    return _wrapped


def _default_hash(key: str) -> int:
    return int(sha256(key.encode()).hexdigest(), 16)


class FileShortKey(File):
    """Like partd's File Interface, except the filenames have to be short
    so we shorten the keys first with a hash. We basically pray that we won't
    have a hash collision.

    We also allow ourselves to change the contents of a file.

    """

    # noinspection PyShadowingBuiltins
    def __init__(self, path=None, dir=None,
                 hashfunc: Callable[[str], int] = _default_hash):
        self._hash = hashfunc
        super().__init__(path, dir)
        self.lock = NotALock()

    def __getstate__(self):
        state = super().__getstate__()
        state.update({'hash': self._hash})
        return state

    def __setstate__(self, state) -> None:
        self._hash = state['hash']
        super().__setstate__(state)

    def append(self, data, *args, **kwargs) -> None:
        """Append a dictionary-like data to the partd.

        """
        new_data = {self._shortkey(k): v for k, v in data.items()}
        super().append(new_data, *args, **kwargs)

    _get = _shorten_second_arg(File._get)
    _iset = _shorten_second_arg(File._iset)
    iset = _iset
    _delete = _shorten_second_arg(File._delete)
    filename = _shorten_second_arg(File.filename)

    def _shortkey(self, key: str) -> str:
        return base_repr(self._hash(key), 32)


class UpdatingClient(Client):
    """A Client that lets you change something after you set it once.

    """
    iset = Client._iset


def file_cache(path: Path) -> Interface:
    """Gives a partd interface to a given filesystem path. Uses

    Parameters
    ----------
    path - Path to the directory based database.

    """
    return Pickle(FileShortKey(path))


def client_cache(address: str) -> Interface:
    """Gives a partd interface to a given partd server by address.

    Parameters
    ----------
    address - Address to the partd server.

    """
    return Pickle(Client(address))


def server_cache(partd: Interface) -> Server:
    """Gives a server address to a partd server cache for future clients.

    Parameters
    ----------
    partd - Underlying interface to use for saving the data.

    """
    return Server(hostname=get_ip_interface('ib0'),
                  partd=partd)


OptoOpFunc = Callable[[OperationalState], OperationalState]
OptoOp = Union[OptoOpFunc, partial]


def cached(func: OptoOp, cache: Interface) -> OptoOpFunc:
    """A Decorator factory that turns functions and a cache factory into
    a function that caches its results in the cache.

    Parameters
    ----------
    func - Function to cache
    cache - Factory that generates an interface to a cache.

    """
    def _new_func(state: OperationalState) -> OperationalState:
        key = get_key(state)
        try:
            return cache.iget(key)
        except KeyError:
            results = func(state)
            cache.iset(key, results)
            return results
    return _new_func
