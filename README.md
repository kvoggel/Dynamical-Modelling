# Dynamical-Modelling

The code sersic2mge provides a Multi-Gaussian Expansion (MGE) of any Sersic function. It's similar to what is provided in the mge-fit (https://pypi.org/project/mgefit/) package but can use Sersic profiles as inputs rather than just the source image. It requires the mge-fit package as a dependency.

Enabling e.g. that one can fit a galaxy with multiple components (e.g. Bulge and Disk), anf get a ME for each component and thus also a mass model for each of those components. These MGEs can then be used in dynamical models (e.g. Jampy also from Michele Cappelari) to model the dynamics of a galaxy.
