import archibald.numpy as np
from archibald.modeling.interpolation_unstructured import UnstructuredInterpolatedModel
import matplotlib.pyplot as plt

"""
Example script demonstrating the use of UnstructuredInterpolatedModel in Archibald.
This class is used when data is "unstructured" (point cloud) rather than on a grid.
It works by resampling the data onto a structured grid internally using RBF interpolation.
"""

print("--- Unstructured Interpolation Example ---")

# 1. Generate some unstructured (randomly spaced) data
# Function: z = x^2 + y^2
n_points = 150
x_rand = np.random.uniform(-3, 3, n_points)
y_rand = np.random.uniform(-3, 3, n_points)
z_rand = x_rand**2 + y_rand**2

x_data = {
    "x": x_rand,
    "y": y_rand
}

# 2. Create the unstructured interpolator
# We specify how many points to use for the internal structured grid resampling
interp = UnstructuredInterpolatedModel(
    x_data=x_data,
    y_data=z_rand,
    x_data_resample=30, # Resample to a 30x30 grid
)

# 3. Evaluate on a regular grid for plotting
x_plot = np.linspace(-3, 3, 50)
y_plot = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x_plot, y_plot)

# Reshape grid for evaluation
Z_interp = interp({
    "x": X.flatten(),
    "y": Y.flatten()
}).reshape(X.shape)

# 4. Plot results
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot original unstructured points
ax.scatter(x_rand, y_rand, z_rand, color='black', alpha=0.5, label="Data Points")

# Plot interpolated surface
surf = ax.plot_surface(X, Y, Z_interp, cmap='viridis', alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("Unstructured Interpolation (Point Cloud to Surface)")
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()
