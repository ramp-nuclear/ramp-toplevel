"""Configuration of this package in general.

"""
import logging
from datetime import date
from functools import partial
from pathlib import Path
from tempfile import mkdtemp

import dask
import yaml
from dask_jobqueue import SLURMCluster

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent / "ramp.yaml"
dask.config.ensure_file(source=str(CONFIG_PATH))


def make_logging_directory(path) -> str:
    """Make a directoy to put the logs in for everything.

    Parameters
    ----------
    path - Path to the directory to put them in.

    """
    p = Path(path).expanduser().absolute()
    p.mkdir(exist_ok=True, parents=True)
    return mkdtemp(dir=str(p))


LOG_DIRECTORY = make_logging_directory(f'~/.ramp/{date.today()}')

with CONFIG_PATH.open() as f:
    defaults = yaml.load(f, Loader=yaml.Loader)
    mc = defaults['jobqueue']['mc']
    burnup = defaults['jobqueue']['burnup']
    mc['log-directory'] = burnup['log-directory'] = LOG_DIRECTORY

dask.config.update(dask.config.config, defaults, priority="new")

BurnupCluster = partial(SLURMCluster, config_name='burnup')
