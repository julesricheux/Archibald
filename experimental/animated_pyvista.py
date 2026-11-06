# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 00:38:27 2026

@author: jules

Animated potential flow past a sphere (dark theme) with PyVista.

pip install pyvista numpy
"""
import numpy as np
import pyvista as pv

U, A = 1.0, 1.0  # free-stream speed, body radius


def velocity(p):
    """Analytic potential flow (uniform stream + doublet) around a sphere, flow along +x."""
    r = np.linalg.norm(p, axis=-1, keepdims=True)
    ex = np.array([1.0, 0.0, 0.0])
    return U * (ex + A**3 / 2 * (ex / r**3 - 3 * p[..., :1] * p / r**5))


# --- Streamlines: RK4 integration from a grid of seeds upstream --------------
g = np.linspace(-1.8, 1.8, 15)
Y, Z = np.meshgrid(g, g)
seeds = np.c_[np.full(Y.size, -4.0), Y.ravel(), Z.ravel()]
seeds = seeds[np.hypot(seeds[:, 1], seeds[:, 2]) < 1.8]

n_steps, dt = 4*220, 0.04/4
paths = np.empty((len(seeds), n_steps, 3))
p = seeds.copy()
for i in range(n_steps):
    paths[:, i] = p
    k1 = velocity(p)
    k2 = velocity(p + dt / 2 * k1)
    k3 = velocity(p + dt / 2 * k2)
    k4 = velocity(p + dt * k3)
    p = p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Build one polyline per streamline, then thicken into tubes
n, m = paths.shape[:2]
cells = np.hstack([np.full((n, 1), m), np.arange(n * m).reshape(n, m)]).ravel()
lines = pv.PolyData(paths.reshape(-1, 3), lines=cells)
lines["speed"] = np.linalg.norm(velocity(lines.points), axis=1)
tubes = lines.tube(radius=0.01)

# --- Body coloured by pressure coefficient Cp = 1 - |v|^2 / U^2 --------------
body = pv.Sphere(radius=A, theta_resolution=80, phi_resolution=80)
body["Cp"] = 1 - (np.linalg.norm(velocity(body.points), axis=1) / U) ** 2

# --- Moving particles --------------------------------------------------------
rng = np.random.default_rng(0)
offsets = rng.integers(0, n_steps, size=len(seeds))  # random start along each streamline
idx = np.arange(len(seeds))
particles = pv.PolyData(paths[idx, offsets].copy())

# --- Scene -------------------------------------------------------------------
pl = pv.Plotter(window_size=(540, 960))
pl.set_background("#050818")  # flat colour: gradients can break shaders on some GPUs
pl.add_mesh(body, scalars="Cp", cmap="turbo", smooth_shading=True, show_scalar_bar=False)
pl.add_mesh(tubes, scalars="speed", cmap="viridis", opacity=0.35, show_scalar_bar=False)
pl.add_mesh(particles, color="white", point_size=7, render_points_as_spheres=False)
pl.add_axes(color="white")
pl.camera.zoom(1.3)

pl.show(interactive_update=True, auto_close=False)  # open window without blocking

running = True


def stop(*_):
    global running
    running = False


# Fires when the window is closed (X button) or 'q' is pressed
pl.iren.interactor.AddObserver("ExitEvent", stop)

frame = 0
while running and not pl._closed:
    frame += 1
    particles.points[:] = paths[idx, (offsets + frame) % n_steps]  # in-place update
    particles.points = particles.points                            # flag points as modified
    pl.update(stime=5, force_redraw=True)

if not pl._closed:
    pl.close()  # release the GL context cleanly

# To export a GIF instead of the live window:
#   pl = pv.Plotter(off_screen=True); ...; pl.open_gif("flow.gif")
#   for i in range(n_steps): particles.points = paths[:, i]; pl.write_frame()
#   pl.close()