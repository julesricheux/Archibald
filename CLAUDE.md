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
- The default environment is fresh water in practice (about 999.13 kg/m³), though `Environment` defaults to 35 g/kg of salinity: `Water.density` divides the salinity by 1000 before a formula that expects g/kg (UNESCO form, about 1026 kg/m³ at 35 g/kg and 15 °C), so salinity adds only 0.03 kg/m³. Not fixed: fixing it moves every hydrostatic and hydrodynamic result by about 2.7 %.

## Sailboat torsors

Every `compute_*` stores results in `boat.forces` and `boat.moments`: `F<x>_<i>` / `M<x>_<i>` per element, `F<x>` / `M<x>` in total. Keys: `Fw` weight, `Fb` buoyancy, `Fh` hull hydrodynamics (resistance, lateral force, dynamic lift, added resistance in waves), `Fi` hull interaction, `Faa` windage, `Fsail` rigs, `Fapp` appendages, `Fprop` propellers, `Ftot` total. `Fh` includes `Fi` when `compute_resistance` computed the interaction; `compute_torsor` sums elements, so it does not count it twice.

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

- `compute_torsor(op_point, components=None, exclude=None)` takes groups (`w`, `b`, `h`, `i`, `aa`, `sail`, `app`, `prop`) or elements (`w_1`, `aa_crew`). Default is all that was computed. It lists what it summed in `boat.torsor_elements`. Typos, components asked for but not computed, and included groups computed with another `OperatingPoint` object all raise.
- `find_equilibrium` and `equilibrium_residuals` are static and take `Ftot`, `Mtot` explicitly, so external terms go in as `Ftot + Ffoil`. Call `find_equilibrium` once per `Opti`, since each call adds its constraints.
- Bounds are ordinary constraints that iterates can violate. Pass `detect_simple_bounds=True` when leaving them is catastrophic, such as a negative speed in a resistance polynomial.

VLM:
- `mesh_geometry` applies `op_point.apply_transformations` to the panels and strips, so results follow heel, trim, leeway and immersion, in the motion frame. `Sailboat` passes `xyz_ref=[0, 0, 0]`, so moments are about the motion frame origin like every other torsor. `draw_vlm` draws moved copies of the hulls.
- `alpha` is read in each strip's section plane, `arctan2(stream . strips_z, stream . strips_x)`, so it works for horizontal foils (`test_horizontal_foil_lift_follows_pitch`). It used to be read in the boat x-y plane, which gave about 0 on a horizontal strip whatever the pitch; vertical strips are unchanged and raked sails moved by under 0.1 %.
- Horizontal foils: a symmetric section gives the same lift in either section order. A cambered one lifts up when its sections are listed port to starboard.
- Free surface at z = 0: every panel force (`panels_density`, at the bound vortex centre) and strip force (`strips_density`, at the quarter chord) uses the water density below and the air density above, whichever the analysis, blended by `water_fraction(z) = sigmoid(-FREE_SURFACE_BETA z)`, 99 % at `FREE_SURFACE_TRANSITION` = 1 cm (`vortex_lattice_method.py`). The circulation solve, the freestream (apparent wind for sails, boat speed for appendages) and the NeuralFoil Reynolds number (analysis fluid viscosity) are unchanged. `pressure_coef` divides by the local density, so it does not jump at the surface. The slope is about 1.1e5 kg/m^4 at z = 0, steep for optimizers. On the Nacra, 3 of the 27 appendage strips are in air, and the appendage side force went from 3406 to 2975 N.
- Strip lift goes along `strips_lift_axis`, perpendicular to the strip freestream and on the side of `strips_z` (the section normal). The force along the flow is then only profile plus induced drag (`test_lift_axis.py`).
- NeuralFoil results are not stored on the VLM: substituting a solution in them took minutes and ended in `std::bad_alloc`.

Weight:
- `Sailboat(masses=[MassProperties(...), ...])`, computed by `MassProperties.compute_weight(op_point)`. CoGs are boat-fixed coordinates, and mass and CoG can be `Opti` variables.
- `displacement` and `cog` still work and act as a single mass when `masses` is empty. Passing `masses` with `displacement` raises `ValueError`.
- A boat with neither has zero weight, with no warning.
- `sum(boat.masses)` gives total mass, CoG and inertia.

Hull hydrodynamics:
- `compute_resistance(op_point, method)` on `Hull` and `Sailboat` holds every hull force: resistance, lateral force, dynamic lift and added resistance in waves. `method` is a key of `DEFAULT_RESISTANCE_METHODS` in `hull.py`, a dict of components, or a list of them, e.g. `["dsyhs", "kijima", "mittendorf"]`. A component given twice raises. `**kwargs` go to every function, so a name means the same thing everywhere (`trim` is Savitsky's keel trim, bow up).
- The component name sets the direction, in the motion frame: `R...` along -x at `cow` (`Ri` amidships, with its lift), `L` along +y amidships, `Fz` along +z at `cow` or the method's point, `N` a yaw couple. Other names raise, `My` included: a pitching moment comes from the point of `Fz`. A component can be `(function, point_function)`, the point in the hull frame, e.g. `savitsky.point_Fz_savitsky` (centre of pressure from `app`).
- Dry hulls: below `DRY_VOLUME` (1e-9 m³) of immersed volume, `Hull.is_wet()` is false, the methods get `dry_reference_hydrostatics()` (the hull upright, immersed to half its mesh depth) and their result is replaced by zero. Evaluated on a dry hull, DSYHS `Rw` gave 1.7e16 N, and wrapping only the output in `np.where` still left NaN in CasADi Hessians (IPOPT "Invalid number" or restoration failure). `test_dry_hull_has_zero_force_and_finite_derivatives` checks it.
- Points are hull-frame, then moved with the operating point, so under attitude their x changes the moment of a force along x. The hull stores `resistance_components` (`_total` sums only `R...`), `resistance_points` (motion frame) and `resistance_torsors`. Multihull interaction reads `Rf` and `Rw`.
- Methods: calm water `dsyhs`, `holtrop`, `michell`, `savitsky` (dynamic part only); waves `savitsky_brown`, `mittendorf` (needs `swh`, `tp`, `Le`, `Lr`, JONSWAP/PM spectrum, `wave_heading` 180 in head seas); leeway `jones`, `inoue`, `kijima`, whose functions take `V` [m/s] and `beta` [rad]. +leeway gives +y and -x, like a fin. Lateral methods alone are allowed. Each method's assumptions and sources are in its module. Keep them sourced: coefficients found only from memory stay out.
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

- Panels are coloured by `vlm.pressure_coef`, with the hydro sign flipped so that appendages read like sails. `pressure_coef` itself is unchanged.
- `animate` defaults to on with pyvista, and the window stays open until closed or `q`. Plotly is static.
- Particles follow the local VLM velocity (`Sailboat._streamlines_travel_time`), limited to 0.1-3 x free stream because the wake filaments are singular. They tend to `op_point._stw` in water and `op_point._aws(z)` in air.
- Default seeds start upstream, one per leading-edge panel, moved along the local freestream: to the upstream edge of the drawn water plane for appendages, and one reference chord ahead of the foremost sail point along the apparent wind for sails. `streamlines_length` then counts from the leading edges, with the upstream distance added and the step count raised to keep the step size (capped at 2000). `seed_fn=Sailboat.trailing_edge_seed_points` gives the old wake-only seeds.
- `draw_forces=True` draws the torsor forces as arrows (pyvista and plotly), tail at the application point, label `name: value unit` at the tip (`force_unit` N, kN, kgf, tf). `force_split=True` or a list of groups (`["b", "sail"]`) gives one arrow per element, otherwise one per group. Elements default to `torsor_elements`, else all computed. `get_force_arrows` gives the arrows: the point is on the exact line of action of (F, M), closest to a reference (CoG, hull `cob`, wing panel centroid, propeller centre; force-weighted mean for a group). The moment along F has no line of action and is returned as `couple` (1.1 kN.m on the Nacra rig). Forces under 1e-6 of the largest are not drawn. Labels are added one call each, because one `add_point_labels` call drops overlapping labels.
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
- **Test baseline: 363 passed, 16 failed, 4 errors.** Compare against this rather than expecting green:
  - 16 failures and errors use the removed old API (`arb.Airplane`, `LiftingSet`, `VortexLatticeMethod`).
  - 3 are Kulfan similarity tests (NaN in `airfoil_families.py`).
  - 1 is the black box nonsmooth solver test.
- `Sailboat` and `draw_vlm` have almost no unit tests (only `geometry/test_force_arrows.py`). Check them with a script.
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

- VLM: `spanwise_resolution` > 1 raises `IndexError` in `viscous_computation` (`strips_y` has one row per section pair). Use more sections instead.
- `ThinAirfoil` with `mc=0` is not symmetric through NeuralFoil: CL is +0.77 at +8 deg and -0.28 at -8 deg. Use a section with thickness, such as `Airfoil("naca0012")`, for appendages.
- 49er: `examples/02 - Geometry/49er_sailboat_object.py` balances Fx and Fy (stw, leeway) at heel 0 and dz 0, with one placeholder mass of 200.5 kg: 18.6 kt and -2.24 deg at TWA 50, TWS 20, in 5 IPOPT iterations from (18 kt, -2.5 deg). From (12 kt, -5 deg) it does not converge in 50. The rig heeling moment (-11.5 kN.m) is not balanced, so the speed is that of an upright boat.
- Molenez load cases: `molenez_hybrid.py` and `molenez_gz_curve.py` read the columns of the load case table the user provided as displacement [t], LCG, TCG, VCG [m] in mesh axes. This is not confirmed.
- `molenez_hybrid.py` (sails, fins, rudder, propeller; 5 dofs, Mz free) converges in 6 iterations at STW 8, TWA 60, TWS 8, containers case: T 1.319 m, heel -6.0 deg, leeway -1.7 deg, 172 rpm, 1.03 kN of thrust. At TWS 15 the wings overdrive the boat and there is no solution with rpm > 0. The propeller, wake fraction and thrust deduction are placeholders. Its fins and rudder are planform outlines, read with `geom_utils.outline_to_le_chords`.
- `molenez_gz_curve.py`: heel sweep 0-90 deg with T and trim solved (Fz, My), all converged. For the containers case, GZ max is 0.89 m at 32 deg and stability vanishes at 82 deg. The hull is taken as watertight, with no deck edge or downflooding.
- `49er_foiling.py`: a placeholder T-foil at the daggerboard tip, Fx and Fz (stw, dz) under fixed thrust. With a fixed incidence the dry hull makes the problem singular, because nothing depends on dz, so the foil incidence falls with dz like a wand. The thrust sweep (50-450 N) jumps from 12.6 to 21.7 kt between 275 and 300 N, at the take-off hump (297 N at 15 kt, 255 N at 17 kt). A fixed-speed sweep follows the curve continuously through lift-off. Every solve converges, but only with 25 sections on the strut and rudder. With 10, the strips leaving the water one by one through the 1 cm density blend made the drag non-monotonic, and 2 thrusts failed.
- Every Nacra 15 figure below predates the VLM lift axis fix (sail drive about doubled on the 49er), so rerun before trusting them.
- Nacra 15: `examples/02 - Geometry/nacra15_equilibrium.py` balances Fx and Fy (stw, leeway) at fixed heel, with dz an `opti.parameter`. Its DRAFT section, commented out, solves dz from Fz instead (-0.011 m for the 250 kg of placeholder, unsourced masses at heel -8, windward hull just wet). The 3-dof balance with Mz has not converged on any set of dofs tried. Yaw cannot balance at a sensible leeway: the sail force acts at x about 2.1 m, ahead of the fin (1.5 m), and the rudders sit at x about 0, so they move Fy (about 280 N/deg at 12 kt) but not Mz. Balance then needs the fin to carry about 1.4 times the sail side force and the rudders to push to leeward. After the VLM orientation fix (TWA 55, TWS 20, heel -8, trims as in `nacra15_resistance.py`), a damped numerical Newton on (stw, leeway, rudder) for (Fx, Fy, Mz) kept Mz above 1.2 kN.m; with Jones hull side force added, Mz only neared zero at about 16 deg leeway and 5 kt. Check the longitudinal positions of the rig and boards first. Rudder angle there: rotation about the leading-edge line, bottom to top, positive pushes to port.
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
- `michell.py`: `S_MAX = 1` clips the calibration to 1, so `compute_dRw_michell(calibrate=True)` has no effect unless `s_max` is raised. The clip now acts on `s**2 = Rw / R_ii` before the square root, because a slightly negative empirical `Rw` (DSYHS, -0.03 N on a hull leaving the water) made the interaction NaN. The truncation at `depth_decay=16` costs 0.1 to 0.5 % (1.2 % at Fn 0.15). The default `n_u=401` aliases at large separations (5 % at S/L 50).
- `compute_Raw_dsyhs`: 7 table entries are zero, and interpolating towards them gives negative resistance (51 of 9430 points, -784 N at worst). The table's source is not referenced.
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
- Lateral force and added resistance in waves were merged into `compute_resistance`: `compute_lateral_force`, `DEFAULT_LATERAL_METHODS` and the `lat` group are gone. Results were checked equal to the old `Fh + Flat` (4.6e-16). Before, a custom dict with `Fz` (the 49er examples) summed the lift into the resistance along -x; it is now vertical. `mittendorf.py` was rewritten with `np.where` (equal to the published code to 4.6e-16, CasADi gradients checked) and gained the irregular-sea `compute_Raw_mittendorf`; the `mm_raw` docstring had swapped "lower" and "upper".
- VLM lift axis fixed: the strip axes were [freestream x, span y, `strips_z`], but `strips_z` is the section normal, perpendicular to the chord. The basis was oblique by alpha, so the NeuralFoil lift leaned back and added a drag of about L sin(alpha): 309 of the 496 N of drag of the 49er boards at 10 kt and -7.2 deg, and about half of the sail drive. With it, the 49er had Fx < 0 on the whole Fy = 0 curve from 5 to 16 kt, so no equilibrium existed, which first looked like a solver problem. The axis commented out in the code, `cross(freestream_x, y)`, is now used with the sign of `strips_z`.
- VLM section order fixed: NeuralFoil's alpha and CL use a fixed boat-frame sign, but the strip lift went along the section frame normal, which follows the order of the sections, and the induced angle `arctan2(Di, L)` jumped to about 180 deg for a negative lift. A vertical fin then gave Fy of opposite signs depending on the section order (-2458 against +16496 N in an earlier test), and the Nacra fin pushed to leeward at zero leeway (+2975 N, now -67 N). `strips_z` is now oriented along chord x z_boat (`strips_orientation`), `alphai = arctan(Di L / (L^2 + eps))`, and NeuralFoil gets the airfoil flipped where the normal was flipped, so camber follows the panel mesh (upper side on the section frame normal). Sails were unchanged on the Nacra. `test_section_order.py` checks it. With sections listed bottom to top, `Airfoil(...).scale(scale_y=-1)` puts the camber lift of a vertical fin towards -y (starboard).
