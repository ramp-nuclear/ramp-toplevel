"""Utilities for running the external subpackages.

"""
import tempfile


class TemporaryDirectory(tempfile.TemporaryDirectory):
    """
    like tempfile.TemporaryDirectory but with an
    optional argument to not clean directory afterwards.
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, suffix=None, prefix=None, dir=None, clean_dir=True):
        self.clean_dir = clean_dir
        if self.clean_dir:
            super(TemporaryDirectory, self).__init__(suffix=suffix, prefix=prefix, dir=dir)
        else:
            self.name = tempfile.mkdtemp(suffix, prefix, dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.clean_dir:
            super(TemporaryDirectory, self).__exit__(exc_type, exc_val, exc_tb)
