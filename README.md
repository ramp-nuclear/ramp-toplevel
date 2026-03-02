Welcome to the high-level analyses package for the ramp project!

This package deals with top-level analyses such as finding the end of
a cycle, performing multiple-state analyses such as rod worths and the
like.

Installation
============

Currently, installation is a bit of a mess.
You need to have an environment with all of our internal tools
installed.
These include ramp-core, isotopes, reactions, coremaker, coreoperator,
corecompute, endf, and batman.

Most of those are easily installed with a "pip install ." command in
their respective directories, except for the ENDF package which is a bit
of a mess compared to the others in this regard. For that package, you
would need to run the "install.sh" script therein.

Dependency installation can mostly be done out of order, but let us know
if you encounter some issues.

Once installed, for some reason we found errors trying to just install
this package with "pip install .", and it seems that one has to update
their conda environment with requirements.yml file here directly before
installation.
We don't know why this happens yet. Sorry.

