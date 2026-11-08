# Templates: your boat in Archibald

These files are starting points for your own project, not ready-to-run examples. Each one sets out the structure of a complete equilibrium for one type of boat. You replace the geometry, masses and settings with your own, then run it.

| Template | Boat | What is solved |
|---|---|---|
| [`motor_ship.py`](motor_ship.py) | Displacement motor ship (cargo vessel, trawler, ferry) | Propeller rpm, draft and trim at a given speed |
| [`monohull_sailboat.py`](monohull_sailboat.py) | Cruising monohull | Speed, leeway, heel and draft in a given wind |
| [`multihull_sailboat.py`](multihull_sailboat.py) | Cruising trimaran or catamaran | Speed, leeway, heel and draft in a given wind |
| [`planing_motorboat.py`](planing_motorboat.py) | Planing boat (RIB, runabout) | Thrust, draft and trim at a given speed |
| [`foiling_motorboat.py`](foiling_motorboat.py) | Motorboat on foils (RIB with U-foils) | Thrust, ride height and trim at a given speed |
| [`foiling_racing_sailboat.py`](foiling_racing_sailboat.py) | Racing monohull with foils and canting keel (IMOCA, Mini 6.50) | Speed, leeway, heel and ride height, for a given cant angle |

For working examples with real geometry and checked results, see [`../02 - Geometry`](../02%20-%20Geometry/). The Molenez files there show a cargo sailboat, the 49er files a skiff, and `49er_foiling.py` a foiler taking off.

## How every template is organised

The same nine steps appear in every file, so once you know one, you know them all:

1. **Settings**: the conditions you want to study (speed, wind) and the files of your geometry.
2. **Hull(s)**: a closed mesh of each hull (STL), in boat axes.
3. **Appendages**: keels, rudders and foils, each described by a few sections (leading edge point and chord).
4. **Rig or propulsion**: sails as wings, propellers, or a simple thrust.
5. **Masses**: every weight with its centre of gravity. This is what the boat has to float and carry.
6. **The boat**: the parts assembled into a `Sailboat` object (the name is historical: it also holds motor boats).
7. **Unknowns**: what Archibald should find (speed, draft, heel...), each with a first guess.
8. **Forces and equilibrium**: which forces are computed, and which ones must balance.
9. **Results and drawing**: printed numbers and a 3D view.

## What you need

- **Hull mesh**: a closed STL of each hull, exported from your CAD or lines program in metres. Sharp leaks or open edges give wrong volumes. A few thousand faces are usually enough; hydrostatics depend a little on the mesh, so check the displacement at your design waterline against your own hydrostatics.
- **Appendage and sail shapes**: for each one, a few sections from root to tip: the leading edge point (x, y, z) and the chord. You can type them in the template, or read them from a DXF with `dxf_to_le_chords` or `outline_to_le_chords` from `archibald.toolbox.geom_utils`, as in the examples.
- **Masses**: lightship and loads, each with mass and centre of gravity. One total mass is enough to start with.
- **Sections (airfoils)**: names such as `"naca0012"` or `"sd7037"`, or `ThinAirfoil(xc, mc)` for a soft sail.

## Axes and signs

Every position is in boat axes, in metres:

- **x** points forward, **y** to port, **z** up. The origin is yours to choose. The transom at the waterline or the keel is common.
- **Heel** is positive to starboard, **trim** positive bow down, **leeway** positive to starboard.
- `dz` moves the boat up: a hull drawn with its keel at z = 0 floats at draft T with `dz = -T`.
- Speeds are in knots, everything else in SI units: m, kg, N, N.m.
- The water is sea water by default (15 °C, 35 g/kg, about 1026 kg/m³).
- A true wind angle `twa` > 0 means wind from starboard, so the boat heels to port (negative heel).

## How the equilibrium works, in short

Archibald writes every force as a formula of the unknowns, then asks an optimizer (IPOPT, through CasADi) to find the values for which the chosen forces and moments add up to zero. Three habits make this easy:

- **Start with few unknowns.** Fix the heel and the draft first and solve only speed and leeway, then free the others one at a time.
- **Give good first guesses.** The solver starts from the `init_guess` of each unknown. A guess far away from the answer may not converge.
- **Check the answer.** An equilibrium is not always a stable one: heel the boat a little more and check that the moment brings it back.

## Before trusting the numbers

Every method in Archibald comes from a published source, given in its module, and has a range of validity: hull type, Froude number, proportions. [`../03 - Validation/README.md`](../03%20-%20Validation/README.md) says which ones have been compared with published data. Results outside a method's range, or from a model not yet validated, are a first estimate, not a prediction.

If something looks wrong, please say so: see [`CONTRIBUTING.md`](../../CONTRIBUTING.md). Reports from naval architects are what make the package better.
