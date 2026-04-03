# -*- coding: utf-8 -*-

from pathlib import Path

_archibald_root = Path(__file__).parent

__version__ = "2.1.0-alpha"
__meet_Archibald__ = "https://https://en.wikipedia.org/wiki/Captain_Haddock"
__documentation__ = None # TODO

from archibald.common import *
from archibald.optimization import *
from archibald.modeling import *
from archibald.geometry import *
from archibald.environment import *
# from archibald.weights import *
from archibald.performance import *
from archibald.dynamics import *
# from archibald.aerodynamics import *

try:
    from importlib.metadata import version

    __version__ = version("archibald")
except Exception:
    __version__ = "unknown"


def docs():
    """
    Opens the archibald documentation.
    """
    import webbrowser

    webbrowser.open_new(
        "https://github.com/julesricheux/archibald/tree/main/archibald"
    )  # TODO: make this redirect to a hosted ReadTheDocs, or similar.


def run_tests():
    """
    Runs all of the Archibald internal unit tests on this computer.
    """
    # TODO implement tests
    # try:
    #     import pytest
    # except ModuleNotFoundError:
    #     raise ModuleNotFoundError(
    #         "Please install `pytest` (`pip install pytest`) to run archibald unit tests."
    #     )

    # import matplotlib.pyplot as plt

    # with plt.ion():  # Disable blocking plotting
    #     pytest.main([str(_archibald_root)])