from pathlib import Path

from setuptools import find_packages
from conda_setup import setup

dirpath = Path(__file__).parent

with (dirpath / 'README.md').open('r') as f:
    long_description = f.read()

if __name__ == '__main__':
    setup(name='ramp',
          description="Reactors Analysis Managment Program",
          long_description=long_description,
          long_description_content_type="text/markdown",
          package_data={'': ['ramp.XXXX']},
          packages=find_packages(),
          scripts=[],
          entry_points={},
          classifiers=[
              "Programming Language :: Python :: 3",
              "Operating System :: OS Independent",
          ],
          python_requires='>=3.10',
          requirements_yml='requirements.yml',
          )

