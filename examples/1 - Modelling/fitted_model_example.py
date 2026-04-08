import archibald.numpy as np
from archibald.modeling.fitting import FittedModel
import matplotlib.pyplot as plt

"""
Example script demonstrating the use of the FittedModel class in Archibald.
FittedModel allows fitting an analytical model to n-dimensional unstructured data
using an automatic-differentiable optimization approach (via CasADi).
"""

def run_1d_example():
    print("--- 1D Fitting Example ---")
    
    # 1. Generate some noisy data
    # Goal: Fit y = a * x^2 + b * x + c
    true_params = {"a": 2.5, "b": -1.2, "c": 5.0}
    
    x_data = np.linspace(-5, 5, 50)
    y_data_true = true_params["a"] * x_data**2 + true_params["b"] * x_data + true_params["c"]
    
    # Add some Gaussian noise
    y_data = y_data_true + np.random.randn(len(x_data)) * 2.0
    
    # 2. Define the model function
    # The function must have the signature f(x, p) where:
    #   x: the independent variable (array or dict)
    #   p: a dictionary of parameters
    def quadratic_model(x, p):
        return p["a"] * x**2 + p["b"] * x + p["c"]
    
    # 3. Fit the model
    # We provide initial guesses for the parameters.
    fm = FittedModel(
        model=quadratic_model,
        x_data=x_data,
        y_data=y_data,
        parameter_guesses={
            "a": 1.0,
            "b": 0.0,
            "c": 1.0,
        },
        parameter_bounds={
            "a": (0, None), # Force 'a' to be positive
        },
        verbose=False # Set to True to see optimization output
    )
    
    # 4. Results
    print(f"True parameters:   {true_params}")
    print(f"Fitted parameters: {fm.parameters}")
    
    r2 = fm.goodness_of_fit(type="R^2")
    rmse = fm.goodness_of_fit(type="rms")
    print(f"R^2 Score: {r2:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    
    # 5. Prediction
    # We can call the FittedModel object directly like a function
    x_test = np.array([-6, 0, 6])
    y_pred = fm(x_test)
    print(f"Predictions at {x_test}: {y_pred}")
    
    # 6. Plotting (Optional)
    # FittedModel inherits a plot() method from SurrogateModel (works for 1D)
    try:
        print("\nClose the plot window to continue...")
        fm.plot()
    except Exception as e:
        print(f"Could not plot: {e}")

def run_2d_example():
    print("\n--- 2D Fitting Example (Dictionary input) ---")
    
    # 1. Generate data: z = a * x + b * y + c + noise
    n_points = 100
    x_val = np.random.rand(n_points) * 10
    y_val = np.random.rand(n_points) * 10
    
    z_data = 1.5 * x_val + 0.8 * y_val + 2.0 + np.random.randn(n_points) * 0.5
    
    # For multi-dimensional input, x_data must be a dictionary
    x_data = {
        "x": x_val,
        "y": y_val
    }
    
    # 2. Define the model function
    def plane_model(x, p):
        # x is now the dictionary we passed as x_data
        return p["a"] * x["x"] + p["b"] * x["y"] + p["c"]
    
    # 3. Fit the model
    fm_2d = FittedModel(
        model=plane_model,
        x_data=x_data,
        y_data=z_data,
        parameter_guesses={
            "a": 1.0,
            "b": 1.0,
            "c": 0.0,
        },
        verbose=False
    )
    
    # 4. Results
    print(f"Fitted parameters (2D): {fm_2d.parameters}")
    print(f"R^2 Score: {fm_2d.goodness_of_fit(type='R^2'):.4f}")

if __name__ == "__main__":
    run_1d_example()
    run_2d_example()
