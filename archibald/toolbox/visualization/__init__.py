"""
Interactive (Plotly) visualization tools and carpet plot utilities.

Submodules are imported explicitly, so that importing this package needs neither Plotly nor any global plotting
configuration:

    plotly_Figure3D: `Figure3D`, the 3D mesh and line builder used by the `draw(backend="plotly")` methods.
    plotly_utils: Sparsity plots and point clouds. Importing it sets the default Plotly renderer to "browser".
    carpet_plot_utils: Timeouts and NaN handling for parameter sweeps.

Matplotlib tools are in `archibald.toolbox.pretty_plots`.
"""
