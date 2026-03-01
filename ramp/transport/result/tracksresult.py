import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SurfaceTracksResult:
    path: Path
    timestamp: datetime
    filehash: str

    def check_valid(self):
        return self.compute_filehash(self.path) == self.filehash

    @classmethod
    def compute_filehash(cls,path:Path):
        filehash = hashlib.sha256()
        with open(path, 'rb') as file:
            for block in iter(lambda: file.read(128), b''):
                filehash.update(block)
        return filehash.hexdigest()