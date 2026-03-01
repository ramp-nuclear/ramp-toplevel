import pickle
import zlib

from packaging.version import Version

from coremaker.core import Core
from coreoperator import OperationalState
from coreoperator.history.history import History

ZLIB_COMPRESSION_LEVEL = 2 # a sensible default


class FeatherState(OperationalState):
    def __init__(self, *, design_name: str, history: History,
                 release: Version, core: Core):
        super().__init__(design_name=design_name, history=history, release=release, core=core)

    @classmethod
    def from_state(cls, state: OperationalState):
        if isinstance(state, FeatherState):
            return state
        return cls(design_name=state.design_name, history=state.history,
                   release=state.release, core=state.core)

    def new_core(self) -> Core:
        return self.core

    @property
    def core(self):
        return pickle.loads(zlib.decompress(self._core))

    @core.setter
    def core(self, core):
        self._core = zlib.compress(pickle.dumps(core), ZLIB_COMPRESSION_LEVEL)
