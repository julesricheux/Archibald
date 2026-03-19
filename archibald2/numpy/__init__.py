### Import everything from NumPy

from numpy import *

### Overwrite some functions
from archibald2.numpy.array import *
from archibald2.numpy.arithmetic_monadic import *
from archibald2.numpy.arithmetic_dyadic import *
from archibald2.numpy.calculus import *
from archibald2.numpy.conditionals import *
from archibald2.numpy.finite_difference_operators import *
from archibald2.numpy.integrate import *
from archibald2.numpy.interpolate import *
from archibald2.numpy.linalg_top_level import *
import archibald2.numpy.linalg as linalg
from archibald2.numpy.logicals import *
from archibald2.numpy.rotations import *
from archibald2.numpy.spacing import *
from archibald2.numpy.surrogate_model_tools import *
from archibald2.numpy.trig import *

### Force-overwrite built-in Python functions.

from numpy import round  # TODO check that min, max are properly imported
