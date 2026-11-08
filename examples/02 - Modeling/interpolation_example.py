import archibald.numpy as np
from archibald.modeling.interpolation import InterpolatedModel
import matplotlib.pyplot as plt

"""
Example script demonstrating the use of InterpolatedModel in Archibald.
InterpolatedModel maps structured (gridded) N-dimensional data to a continuous function.
It is suitable for use in optimization problems as it is differentiable.
"""

print("--- 1D Interpolation Example ---")

# 1. Generate structured data for a 1D function: y = sin(x)
x_coords = np.linspace(0, 10, 11) # 11 points from 0 to 10
y_data = np.sin(x_coords)

# 2. Create the interpolator
# method="bspline" is differentiable and smooth
interp = InterpolatedModel(
    x_data_coordinates=x_coords,
    y_data_structured=y_data,
    method="bspline"
)

# 3. Evaluate the interpolator
x_fine = np.linspace(0, 10, 100)
y_interp = interp(x_fine)

# 4. Plot results
plt.figure(figsize=(8, 5))
plt.plot(x_coords, y_data, 'ok', label="Original Data Points")
plt.plot(x_fine, y_interp, '-', label="B-Spline Interpolation")
plt.plot(x_fine, np.sin(x_fine), '--', alpha=0.5, label="True sin(x)")
plt.legend()
plt.title("1D Interpolation with InterpolatedModel")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
    
print("\n--- 2D Interpolation Example ---")

# 1. Generate structured data for a 2D function: z = sin(x) * cos(y)
x_coords = np.linspace(0, 5, 15)
y_coords = np.linspace(0, 5, 15)

X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
Z = np.sin(X) * np.cos(Y)

# x_data_coordinates must be a dictionary for N > 1
coords_dict = {
    "x": x_coords,
    "y": y_coords
}

# 2. Create the interpolator
interp_2d = InterpolatedModel(
    x_data_coordinates=coords_dict,
    y_data_structured=Z,
    method="bspline"
)

# 3. Evaluate at a specific point
point = {"x": 2.5, "y": 2.5}
z_val = interp_2d(point)
print(f"Interpolated value at {point}: {z_val:.4f}")
print(f"True value at {point}: {np.sin(2.5) * np.cos(2.5):.4f}")
