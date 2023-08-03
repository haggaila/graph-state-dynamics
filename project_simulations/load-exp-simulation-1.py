# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from project_experiments.error_mitigation import *
from project_simulations.simulation_routines import *

# This file is used to load a dynamical simulation run previously, and plot the results.
# -----------------------------------------------------------------------------
# Plotting and simulation parameters
s_simulation_id = "b2cdcb62bb894c02981b21c191c852fd"  # Set the simulation id to load
b_save_figures = True
fontsize = 16
b_mitigate_readout = True  # Whether to mitigate readout errors for the experiment plots
b_correlations = False  # Whether to plot correlation functions
b_stabilizers = True  # Whether to plot three-qubit stabilizers

plot_experiment_simulation(
    s_simulation_id,
    fontsize=fontsize,
    b_mitigate_readout=b_mitigate_readout,
    b_correlations=b_correlations,
    b_stabilizers=b_stabilizers,
    b_save_figures=b_save_figures,
    b_plot_z=False,
)

plt.show()
tmp = 2  # Put a breakpoint here if desired
