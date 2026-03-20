"""
This module relies on Peter D. Sharpe's awesome work on AeroSandbox.

AeroSandbox
Author: Peter D. Sharpe
Repository: https://github.com/peterdsharpe/AeroSandbox
Version used: 4.2.9
Date retrieved: 2026-03-20

AeroSandbox is distributed under its original MIT license.
All credit for the underlying methods and implementations belongs to the original author.
"""

### Import everything from NumPy

from numpy import *

### Overwrite some functions
from archibald.numpy.array import *
from archibald.numpy.arithmetic_monadic import *
from archibald.numpy.arithmetic_dyadic import *
from archibald.numpy.calculus import *
from archibald.numpy.conditionals import *
from archibald.numpy.finite_difference_operators import *
from archibald.numpy.integrate import *
from archibald.numpy.interpolate import *
from archibald.numpy.linalg_top_level import *
import archibald.numpy.linalg as linalg
from archibald.numpy.logicals import *
from archibald.numpy.rotations import *
from archibald.numpy.spacing import *
from archibald.numpy.surrogate_model_tools import *
from archibald.numpy.trig import *

### Force-overwrite built-in Python functions.

from numpy import round  # TODO check that min, max are properly imported
