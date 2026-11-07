# Archibald

Sailboat performance (VPP) and design library, forked from AeroSandbox. It models sails, hulls and appendages with differentiable, engineering-level physics, so the same code runs numerically or inside an optimizer.

- PyPI name `archibald-vpp`, import name `archibald`, conventional alias `import archibald as arb`.
- Version 2.1.x alpha. Python >= 3.11, developed on 3.13.
- Replaces the old `GEMINI.md` (removed from the repo, last version at `git show 59ab3fc:GEMINI.md`).

## Layout

| Path | Content |
|---|---|
| `archibald/common.py` | Base classes `ArchibaldObject`, `ExplicitAnalysis`, `ImplicitAnalysis`, serialization |
| `archibald/numpy/` | NumPy/CasADi dispatch layer |
| `archibald/optimization/opti.py` | `Opti` and `OptiSol`, wrapping CasADi |
| `archibald/geometry/` | `sailboat.py` (`Sailboat`), `hull.py`, `mesh.py` (`ArchibaldMesh`), `planform.py` (`Rig`, `Appendage`, `Planform`), `propeller.py`, `airfoil/`, `fuselage.py` |
| `archibald/dynamics/aero_3D/` | Vortex lattice: `AeroVortexLatticeMethod`, `HydroVortexLatticeMethod`, lifting lines |
| `archibald/dynamics/aero_2D/`, `aero/` | 2D aerodynamics |
| `archibald/dynamics/hydro/` | Resistance methods (`holtrop`, `dsyhs`, `savitsky`, `mercier_savitsky`, `radojcic`, `michell`, ...) and hull interaction (`multihull`) |
| `archibald/performance/operating_point.py` | `OperatingPoint`: speeds, wind, attitude, and frame transforms |
| `archibald/weights/` | `MassProperties`. Not exported at top level: `from archibald.weights import MassProperties` |
| `archibald/environment/` | Air, water, gravity |
| `archibald/modeling/` | Fitting, interpolation, surrogate models |
| `archibald/toolbox/` | `units`, `mesh_utils`, `statistics/`, `visualization/` (`plotly_Figure3D`, `plotly_utils`, `carpet_plot_utils`) |
| `examples/` | `01 - Modeling`, `02 - Geometry`, `03 - Validation` (literature models against published data, with its own `README.md`) |

`archibald2` and the in-repo `legacy/` are gone. Do not reintroduce them. The sibling folders `../legacy` and `../Experimental` are outside the repo; only read them when asked.

## Conventions

- **Unified math:** use `import archibald.numpy as np` in anything that may run symbolically. Plain NumPy is only for visualization or numeric post-processing, imported as `onp` (see `sailboat.py`).
- **No `aerosandbox` or `asb`:** keep them only in credits and acknowledgements. NeuralFoil imports aerosandbox itself, which is fine.
- **Differentiable first:** no branching on values that may be symbolic, and prefer smooth formulations.
- **Object model:** geometry and analysis objects inherit from `ArchibaldObject`. Physics models inherit from `ExplicitAnalysis` or `ImplicitAnalysis`.
- **Naming:** PEP8 with descriptive names. Use keyword arguments unless self-explanatory, and one parameter per line in long signatures and calls.
- **Mutable defaults:** never in new code; use `None`. Some older signatures in `Sailboat` still have them.
- **Docstrings and type hints:** document purpose, inputs with units in brackets, and returns. The stated standard is Google-style docstrings with `>>>` examples and `X | None` hints, but `geometry/` and `dynamics/hydro/` use NumPy-style docstrings and `Optional`. Match the file you are editing.

## Frames and signs

- Axes: x forward, y port, z up.
- Positive heel is to starboard (+Mx drives it). Positive trim is bow down (+My drives it).
- `OperatingPoint.apply_transformations(geometry, inverse=False, is_vector=False)` goes from boat axes to the boat motion frame: `geom @ mat.T`, then adds `xyz` unless `is_vector`.
- Force and moment results are expressed in the boat motion frame, with moments about its origin.
- Immersion: hull meshes such as Molenez have the keel near z=0, so immersion `T` means `dz=-T`. Water is at z=0 in the motion frame.
- `OperatingPoint` stores `_stw`, `_tws0` and `_aws(z)` in m/s. The public `stw`, `tws0`, `aws(z)` are in knots. Apparent wind varies with height through the wind gradient.
- The default environment is fresh water (about 999.13 kg/m³).

## Sailboat torsors

Every `compute_*` stores results in `boat.forces` and `boat.moments`: `F<x>_<i>` / `M<x>_<i>` per element, `F<x>` / `M<x>` in total. Keys: `Fw` weight, `Fb` buoyancy, `Fh` resistance, `Fi` hull interaction, `Flat` hull lateral force, `Faa` windage, `Fsail` rigs, `Fapp` appendages, `Fprop` propellers, `Ftot` total. `Fh` includes `Fi` when `compute_resistance` computed the interaction; `compute_torsor` sums elements, so it does not count it twice.

- **Shapes:** results are `(1, 3)` "wide" arrays, built with `wide()`. Moments are `np.cross(center, F)`. Index totals as `Ftot[0, i]`.
- **CasADi pitfall:** `(1x3) + (3x1)` raises in CasADi but broadcasts in NumPy, so keep everything wide. Prefer `x = x + y` to `+=` in accumulators.
- **New components:** use `_start_torsor(group, op_point)` and `_add_torsor(group, element, F, M)`. They clear stale keys, and record the operating point checked by `compute_torsor`.

Equilibrium, with `Opti` only and no registry of what varies (`examples/02 - Geometry/molenez_propulsion.py`):

```python
opti = Opti()
op_point = OperatingPoint(stw=8., dz=-opti.variable(init_guess=1.3), heel=opti.variable(init_guess=0.))  # any number can vary
boat.masses[2].y_cg = opti.variable(init_guess=3.)                  # masses
boat.propellers[0].rpm = opti.variable(init_guess=150.)             # propeller, also pitch_ratio, center, axis
rig["mainsail"] = rig["mainsail"].rotate_local(angle_deg=opti.variable(init_guess=-15.), axis=axis, origin=origin)  # wings

boat.compute_weight(op_point)                                       # what is computed
boat.compute_buoyancy(op_point)
boat.compute_propellers(op_point)
Ftot, Mtot = boat.compute_torsor(op_point, exclude=["w_2"])         # what is summed, no physics here

opti.subject_to(boat.propellers[0].performance["BHP"] <= 500e3)     # other constraints and objectives
sol = boat.find_equilibrium(opti, Ftot, Mtot, dofs=["Fz", "Mx", "My"])  # what is balanced, then solve
```

- `compute_torsor(op_point, components=None, exclude=None)` takes groups (`w`, `b`, `h`, `i`, `lat`, `aa`, `sail`, `app`, `prop`) or elements (`w_1`, `aa_crew`). Default is all that was computed. It lists what it summed in `boat.torsor_elements`. Typos, components asked for but not computed, and included groups computed with another `OperatingPoint` object all raise.
- `find_equilibrium` and `equilibrium_residuals` are static and take `Ftot`, `Mtot` explicitly, so external terms go in as `Ftot + Ffoil`. Call `find_equilibrium` once per `Opti`, since each call adds its constraints.
- Bounds are ordinary constraints that iterates can violate. Pass `detect_simple_bounds=True` when leaving them is catastrophic, such as a negative speed in a resistance polynomial.

VLM:
- `mesh_geometry` applies `op_point.apply_transformations` to the panels and strips, so results follow heel, trim, leeway and immersion, in the motion frame. `Sailboat` passes `xyz_ref=[0, 0, 0]`, so moments are about the motion frame origin like every other torsor. `draw_vlm` draws moved copies of the hulls.
- `alpha` is read in the boat x-y plane, so it stays correct under heel but is still poor for near-horizontal strips.
- NeuralFoil results are not stored on the VLM: substituting a solution in them took minutes and ended in `std::bad_alloc`.

Weight:
- `Sailboat(masses=[MassProperties(...), ...])`, computed by `MassProperties.compute_weight(op_point)`. CoGs are boat-fixed coordinates, and mass and CoG can be `Opti` variables.
- `displacement` and `cog` still work and act as a single mass when `masses` is empty. Passing `masses` with `displacement` raises `ValueError`.
- A boat with neither has zero weight, with no warning.
- `sum(boat.masses)` gives total mass, CoG and inertia.

Lateral force:
- `compute_lateral_force(op_point, method)` on `Hull` and `Sailboat`, like the resistance, with `DEFAULT_LATERAL_METHODS` in `hull.py` and functions in `dynamics/hydro/` (`jones`, `inoue`, `kijima`). Components are `L` (normal to the flow), `Ri` (along it) and `N` (yaw moment about amidships). The force is horizontal, and +leeway gives +y and -x, like a fin.
- Added resistance is separate from `compute_resistance`, so it is summed once. Each method's assumptions and sources are in its module. Keep them sourced: coefficients found only from memory stay out.
- Inoue and Kijima are fitted on merchant ships. On the Molenez, the Kijima `N'beta` is negative, so the force acts aft of amidships, against the Munk moment expected of a hull: this comes from extrapolating the regression, so do not trust the yaw moment there.

Hydrostatics:
- `Hull.compute_buoyancy` never moves the mesh. It brings the water plane into the hull frame instead.
- To draw a solved position, move a copy, as in `examples/02 - Geometry/molenez_equilibrium.py`:

```python
boat_sol = sol(boat)
boat_sol.hulls[0].mesh.transform(sol(op_point))
boat_sol.draw()
```

`OptiSol.value` substitutes recursively on a copy. Integer and boolean arrays such as mesh `faces` must keep their type, because pyvista and `ArchibaldMesh.reset_data` index with them.

## draw_vlm

- `animate` defaults to on with pyvista, and the window stays open until closed or `q`. Plotly is static.
- Particles follow the local VLM velocity (`Sailboat._streamlines_travel_time`), limited to 0.1-3 x free stream because the wake filaments are singular. They tend to `op_point._stw` in water and `op_point._aws(z)` in air.
- `particle_speed` is a playback rate in seconds of flow per second of animation (1 is real time, default 0.1), with one clock for air and water.
- `render_points_as_spheres=False` is deliberate: spheres were reported as not rendering on the author's GPU.
- `pv.Plotter(off_screen=...)` is passed explicitly, so it overrides `pv.OFF_SCREEN`.

## Running things

`python` is not on PATH on the author's machine. The environment is `%USERPROFILE%\anaconda3\envs\py313`, described by `environment.yaml`.

```powershell
$env:PYTHONPATH = "$PWD"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:MPLBACKEND = "Agg"
& "$env:USERPROFILE\anaconda3\envs\py313\python.exe" -m pytest archibald -q -p no:cacheprovider --continue-on-collection-errors
```

- `pytest` is not installed in `py313`. `pip install -e ".[test]"` or appending the base conda `site-packages` to `sys.path` both work.
- **Test baseline: 338 passed, 16 failed, 4 errors.** Compare against this rather than expecting green:
  - 16 failures and errors use the removed old API (`arb.Airplane`, `LiftingSet`, `VortexLatticeMethod`).
  - 3 are Kulfan similarity tests (NaN in `airfoil_families.py`).
  - 1 is the black box nonsmooth solver test.
- `Sailboat` and `draw_vlm` have no unit tests. Check them with a script.
- `archibald/numpy/test_numpy/test_all_operations_run.py` calls `np.seterr(all="raise")` when collected, and it applies to every later test of the session: underflows in `softplus` become errors. Tests that hit them need `onp.errstate`, as in `test_validation_published.py`.
- **Validation:** `examples/03 - Validation/` compares literature models with published data and saves figures. Its `README.md` has the status of each model and what is still needed. A CSV of `data/` with a `# TO FILL` header is a blank, and scripts print `[INPUT NEEDED]` until it has rows. Keep sources in each script's docstring. Numbers that must not move go in `test_validation_published.py` and `test_windage_published.py`.
- Examples use relative data paths, so run them from their own folder.
- **CI:** `.github/workflows/run-pytest.yml` runs on `main` and `develop` with Python 3.11-3.13. It excludes the known-broken tests above by name, so when you fix one, remove its entry. CI installs the latest casadi (3.8.2 in November 2026), while `py313` has 3.7.2. IPOPT behaves differently between them, so test a solver test under both: `pip install --no-deps --target <dir> casadi` and put that folder first on `PYTHONPATH`. A new top-level import of an undeclared package breaks `import archibald` for pip users, so declare it in `pyproject.toml`, `requirements.txt` and `environment.yaml`.
- **Publishing:** `publish.yml` runs on `v*` tags and only exists on `main` for now, so tag from `main`. The tag must match the `pyproject.toml` version.
- Optional packages are in the `viz` extra (`pyvista`, `plotly`, `seaborn`, `imageio`). `imageio` is needed for `draw_vlm(gif_path=...)` and is not installed in `py313`. `tqdm`, `cadquery` and non-test `shapely` uses are not declared anywhere.
- For headless rendering, force `off_screen=True` and use `plotter.screenshot`.
- The repo is on a synced SharePoint folder, so `git grep` and broad searches can time out. Scope searches to `archibald/` with a file filter and exclude `archibald/data`.

## Known issues

Confirmed and not fixed unless noted otherwise. Check they still exist before relying on this list.

- VLM, section order: a vertical `Fin` at leeway -3 gives Fy = -2458 N with sections listed bottom to top, but +16496 N with the same sections top to bottom. Not investigated.
- `ThinAirfoil` with `mc=0` is not symmetric through NeuralFoil: CL is +0.77 at +8 deg and -0.28 at -8 deg. Use a section with thickness, such as `Airfoil("naca0012")`, for appendages.
- Nacra 15 sailing equilibrium has not converged on any set of dofs tried. Yaw cannot balance at a sensible leeway: the sail centre of effort (x about 2.2 m) is ahead of the fin (1.5 m) and rudders (0 m), with no hull side force. For speed and leeway, Fx and Fy are nearly tangent at TWA 55 (appendage L/D of only 3 to 5). Not tried on a reach.
- `Hull.compute_resistance`: DSYHS `Rw` is noisy with mesh-derived hull parameters (factor 4 for 1.5 cm of immersion on the Nacra), and changes with leeway through the hull dimensions.
- `compute_propulsion`: the advance speed ignores the angle between flow and shaft, and the torque reaction is not in the moment. Wake, thrust deduction and etaR are inputs, as hulls do not compute them.
- `Sailboat.__init__`: `name` and `xyz_ref` are not stored, and list arguments use mutable defaults.
- `ArchibaldMesh.draw(draw_plane=True)`: `projected_centroid` subtracts one scalar from all three coordinates, so the plane is off-centre in x and y. Its height is correct for a vertical normal.
- Hydrostatics depend on mesh resolution, and not monotonically. On a box at `dz` = 0.2 and 0.3, the volume error is -8% and -11% at 12 faces, +0.8% and -1.2% at 8k, -1.4% and +2.0% at 34k, -0.25% and +0.37% at 135k. Heel differences of about 0.2° are unreliable on the 908-face Molenez mesh, where the upright centre of buoyancy sits about 3 mm off centre.
- `Opti` equilibria only look for balance, so they can land on an unstable or capsized solution when the bounds are wide. Check that a perturbation gives a restoring moment.
- `archibald/__init__.py`: `version("archibald")` looks up the wrong distribution name (PyPI is `archibald-vpp`), so `__version__` likely ends up `"unknown"`. Not run to confirm.
- `compute_Rw_dsyhs`: `smooth_clip(Fr, 0, 0.6, beta=10)` is too soft, so the result leaves the published polynomial, by +131 % at Fn 0.15, -9 % at 0.35 and -26 % at 0.5. `beta=100` matches it within 1 %, untested in solvers. Fixing it moves every Molenez and Nacra result.
- `savitsky.tau`: `softplus(trim, beta=1)` changes the trim (1° gives 1.31°, 2.88° gives 2.94°), so +3 % on lift and +5 % on pressure drag at 2.88°. `beta=10` would remove it.
- `ArchibaldMesh.hydrostatics` smooths with lengths fixed in metres. With one large row of faces touching the waterline from above, the volume error is +10.4 % at L = 1 m, +5.4 % at 10 m, -1.0 % at 100 m. With 12 rows it stays near -0.35 %.
- `michell.py`: `S_MAX = 1` clips the calibration to 1, so `compute_dRw_michell(calibrate=True)` has no effect unless `s_max` is raised. The truncation at `depth_decay=16` costs 0.1 to 0.5 % (1.2 % at Fn 0.15). The default `n_u=401` aliases at large separations (5 % at S/L 50).
- `compute_Raw_dsyhs`: 7 table entries are zero, and interpolating towards them gives negative resistance (51 of 9430 points, -784 N at worst). The table's source is not referenced.
- `mittendorf.mm_raw`: the docstring swaps "lower" and "upper". Output 1 is the upper bound.
- Windage: the drag areas of one crew member are inverted between models. ORC has head 0.271 and beam 0.540 m², Day has head 0.511 and beam 0.305 m².
- `holtrop.py`: `m_1` divides by `T + 0.1` (about -6.6 % on `Rw` at T = 1.3 m, -0.13 % on `m1` at T = 10 m), `compute_Rtr_holtrop` does not use the paper's `Fr_T`, friction and correlation use `Aws - Atr`, and `lcb` from a mesh is a fraction where the method expects percent. The 1984 terms are cited as 1982. There is no `t_single`.
- `miyazawa.py` gives 31 % viscous interference at S/L 0.2, large against the module's own note. Its source has not been read.
- `dynamics/hydro/validation_planing.py` has stale imports and does not run. `savitsky_planing.py` in `03 - Validation` supersedes it.

## History worth knowing

- `archibald2` was renamed to `archibald`, and `aerosandbox`/`asb` were replaced by `archibald`/`arb` throughout.
- Stale `archibald.aerodynamics` and `geometry.airplane` imports were fixed.
- `visualization` moved under `archibald/toolbox/`, with a docstring-only `__init__.py`, so importing the package does not pull in plotly or change matplotlib settings.
- `Sailboat.masses` was added, and `draw_vlm` particles were reworked as described above.
- `compute_torsor` was rewritten to sum without running physics, and `find_equilibrium` to take an `Opti`. `compute_propeller` became `compute_propellers`, built on `Propeller.compute_propulsion`.
- The VLM used to ignore heel, trim, leeway and immersion entirely. Non-zero attitude results changed when this was fixed.
- `multihull_common.interp_nodes` was removed. `jamaluddin.tau_jamaluddin` now uses a 2D bspline `InterpolatedModel` of tau on (Fn, S/L), with `fill_value=None` to hold the edge values. Interpolate tau itself: interpolating A and B of `A (S/L)^B` made a hump of 1.55 at Fn 0.235, S/L 0.2, above every fit. Between the fitted Froude numbers tau changed by up to 0.32 against the old linear interpolation, so `michell_jamaluddin` results move (not rerun on the Nacra).
- `multihull_common.interp_table` was replaced by `table_model`, which returns `factor(Fn, SL)` built on an `InterpolatedModel` and is used by `insel_molland`. It adds a column at 2 max(S/L) holding 1 (no interference), and clamps Fn and S/L to the grid itself, because the numeric linear path (scipy) extrapolates. CasADi's bspline needs at least 4 nodes per axis and raises otherwise, so shorter tables fall back to linear. Below the first S/L the factor is now held, where the old tent functions pulled it toward 0.
- `test_opti_minimize_absolute_value` used `sum(sqrt((x - target)**2 + 1e-6))`, which has almost no curvature away from the target, and diverged under casadi 3.8.2 in CI. It now uses the epigraph form of the L1 norm, `min sum(t)` with `|x - target| <= t`, which solves in 5 iterations from any start. Write nonsmooth objectives in tests this way rather than with a tiny smoothing.
- The worked example of Holtrop & Mennen (1982) found four errors in `holtrop.py`: `c_3` (`RW` was 3.8 % too high), `w_single_open_stern` and `w_twin` (both raised `TypeError`), and the sign in `eta_R_twin`.
