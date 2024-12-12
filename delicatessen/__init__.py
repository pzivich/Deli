"""Delicatessen is a one-stop shop for all your sandwich (variance) needs. Specifically, `delicatessen` implements a
general API for M-estimation. Estimating equations are both pre-built and can be custom-made. For help on creating
custom estimating equations, be sure to check out the ReadTheDocs documentation at
https://deli.readthedocs.io/en/latest/.

To install the delicatessen library, use the following command

$ python -m pip install delicatessen

"""

from .version import __version__

from .estimation import MEstimator, GMMEstimator
from .sandwich import compute_sandwich
