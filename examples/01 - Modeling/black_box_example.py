import archibald.numpy as np
from archibald.modeling.black_box import black_box
from archibald.optimization import Opti
import matplotlib.pyplot as plt

"""
Example script demonstrating the use of the black_box wrapper in Archibald.
This allows you to use functions that do NOT use archibald.numpy (e.g., legacy code, 
external solvers, or math library functions) inside an optimization problem.
Gradients are computed automatically using finite differences.
"""

def legacy_function(x, y):
    """
    A function that uses standard math and does NOT support symbolic computation.
    """
    import math
    # Example: a modified Rosenbrock-like function
    return (1 - x)**2 + 100 * (y - x**2)**2 + math.sin(x)

print("--- Black Box Optimization Example ---")

# 1. Wrap the function
# fd_method can be "forward", "backward", "central", or "smoothed"
wrapped_func = black_box(legacy_function, fd_method="central")

# 2. Set up the optimization problem
opti = Opti()

x = opti.variable(init_guess=0.5)
y = opti.variable(init_guess=0.5)

# Use the wrapped function as the objective
# Note: it supports keyword arguments if the original function does
objective = wrapped_func(x, y)

opti.minimize(objective)

# Add some constraints
opti.subject_to(x >= 0)
opti.subject_to(y <= 2)

# 3. Solve
try:
    sol = opti.solve(verbose=False)
    print(f"Optimal x: {sol(x):.4f}")
    print(f"Optimal y: {sol(y):.4f}")
    print(f"Objective at optimum: {sol(objective):.4f}")
except Exception as e:
    print(f"Optimization failed: {e}")

# 4. Visualization
# Let's see what the function looks like
x_range = np.linspace(-0.5, 1.5, 50)
y_range = np.linspace(-0.5, 1.5, 50)
X, Y = np.meshgrid(x_range, y_range)

# Use vectorization to evaluate the original legacy function for plotting
Z = np.vectorize(legacy_function)(X, Y)

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Objective Value')

# Plot the found optimum
if 'sol' in locals():
    plt.plot(sol(x), sol(y), 'r*', markersize=15, label='Optimum')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization of a Black Box Function')
plt.legend()
plt.show()
