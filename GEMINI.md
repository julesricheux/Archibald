# Archibald (v2.1.x) Project Overview

Archibald is a specialized sailboat performance analysis and design framework. It enables engineers to model and optimize sailboat geometries, including sails (rigs), hulls, and appendages, using differentiable programming and engineering-level physics models.

## Core Technologies
- **Python**: Primary development language.
- **CasADi**: Used for symbolic computation, automatic differentiation, and optimization.
- **archibald.numpy**: A custom bridge that provides a unified interface for both NumPy (numeric) and CasADi (symbolic) operations, allowing the same code to be used for analysis and optimization.
- **Vortex Lattice Method (VLM)**: Employed for both aerodynamic (sails) and hydrodynamic (lifting surfaces) force calculations.
- **Visualization**: Support for PyVista, Plotly, and Matplotlib for 3D/2D plotting.
- **Geometry**: Trimesh for mesh processing and differentiable geometry definitions.
- **Ruff**: Likely used for linting and formatting (indicated by `.ruff_cache` in `.gitignore`).
- **External Binaries**: The project includes `avl.exe` and `xfoil.exe` (in `archibald/`) for external aerodynamic analysis. These are likely called as subprocesses for specific calculations.

## Project Structure
- `archibald/`: The current active development branch (v2.1.x alpha).
  - `data/`: Reference datasets (coefficients, fluid properties, airfoil data).
  - `common.py`: Base classes (`AeroSandboxObject`, `ExplicitAnalysis`, `ImplicitAnalysis`) and serialization.
  - `geometry/`: Sailboat, Hull, Wing, Rig, Propeller, and mesh utilities.
  - `dynamics/`: Aero and Hydro analysis models (VLM, etc.).
  - `numpy/`: The NumPy/CasADi abstraction layer.
  - `optimization/`: Wrappers for CasADi's optimization environment.
  - `performance/`: Operating point definitions and performance metrics.
  - `tools/`: Unit conversions and utility functions.
  - `avl.exe`, `xfoil.exe`: Aerodynamic analysis engines.
- `archibald2/`: Legacy version (v2.0.x). IGNORE IT.
- `legacy/`: Legacy version (v1.x). IGNORE IT.

## Building and Running

### Prerequisites
Inferred dependencies include:
- `casadi`
- `numpy`
- `scipy`
- `dill`
- `trimesh`
- `pyvista` (optional for 3D visualization)
- `plotly` (optional for interactive visualization)
- `pytest` (for running tests)

### Running Tests
The project uses `pytest`. Tests are located within subpackages (e.g., `archibald/numpy/test_numpy/`).
```powershell
# Run all tests
pytest

# Run a specific test suite
pytest archibald/numpy/test_numpy/
```

### Usage
Archibald is intended to be used as a library. Typical usage involves defining a `Sailboat`, an `OperatingPoint`, and running an analysis:
```python
from archibald import Sailboat
# Define components, then assemble a Sailboat
sailboat = Sailboat(...)
# Run analysis
forces = sailboat.compute_sails(op_point)
```

## Development Conventions
- **Differentiability**: Prioritize code that is compatible with CasADi. Use `archibald.numpy` (aliased as `np`) instead of native `numpy` where symbolic support is needed.
- **Object Model**: Most geometry and analysis objects should inherit from `AeroSandboxObject` for consistent serialization and comparison.
- **Analyses**: Differentiate between `ExplicitAnalysis` (direct calculation) and `ImplicitAnalysis` (equations to be solved by an optimizer).
- **Coordinate Systems**: Pay attention to coordinate conventions (often involve X-forward, Y-port, Z-up, but check `sailboat.py` for specific inversions).
- **Documentation**: Many functions contain detailed docstrings. Maintain this standard for new code.

## Critical Development Conventions
- **Unified Math**: ALWAYS use `import aerosandbox.numpy as np`. Never use standard `import numpy`.
- **Optimization-First**: Formulate physics as differentiable functions. Use `opti = asb.Opti()` for problems.
- **Heritage**: Physics models should inherit from `ExplicitAnalysis` or `ImplicitAnalysis`.

## Code Style & Formatting Standards
### 1. Naming Conventions (Strict PEP8)
- **Descriptive Names**: Use `temperature` (not `T`), `wing_tip_coordinate_x` (not `wtx`), `battery_capacity` (not `bc`).
- **Standard Case**: `variable_name`, `def function_name()`, `class ClassName`.

### 2. Layout & Readability
- **Multi-line Grouping**: Split complicated math expressions across lines based on natural groupings.
- **One Parameter Per Line**: In function definitions and calls, place each parameter on a new line unless exceptionally short.
  ```python
  # Encouraged:
  plt.plot(
      time,
      temperature,
      ".-",
      color='lightgrey',
      label="Temperature over time"
  )
  ```
- **Keyword Arguments**: Use them for all functions unless they are self-explanatory (e.g., `asb.Atmosphere(altitude=7000)`).

### 3. Type Hinting & Safety
- **Modern Syntax**: Use `list | dict | tuple` (built-ins) and the `|` operator for unions (e.g., `float | np.ndarray`).
- **Optionals**: Use `X | None` instead of `Optional[X]`.
- **Broad Inputs / Narrow Outputs**: Use `Sequence[T]` or `Iterable[T]` for inputs; specific types for returns.
- **Mutable Defaults**: NEVER use `[]` or `{}` as default values. Use `None` and initialize inside.

## Documentation Requirements
- **Google-Style Docstrings**: Mandatory for all user-facing functions.
- **Content**: Must document purpose, inputs (with types/units in brackets), and returns.
- **Runnable Examples**: Include `>>>` examples in every docstring.

## Reference Example
```python
import aerosandbox.numpy as np

def compute_dynamic_pressure(
    velocity: float | np.ndarray,  # [m/s]
    density: float | None = None,   # [kg/m^3]
) -> float | np.ndarray:
    """
    Computes dynamic pressure for aerodynamic analysis.
    
    Example:
    >>> compute_dynamic_pressure(velocity=100.0, density=1.225)

    Args:
        velocity: Velocity value(s) [m/s] [float or np.ndarray]
        density: Air density. If None, uses sea level [kg/m^3] [float or None]

    Returns:
        Dynamic pressure value(s) [Pa] [float or np.ndarray]
    """
    if density is None:
        density = 1.225
    return 0.5 * density * velocity ** 2
```
