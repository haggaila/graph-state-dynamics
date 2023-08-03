# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from project_simulations.simulation_routines import *
from lindbladmpo.LindbladMPOSolver import *

# This file is used to run the dynamical simulations of the project without corresponding
#    experiments (by specifying the system parameters explicitly), and plot the results.
# -----------------------------------------------------------------------------
# Plotting and simulation parameters
s_unique_id = "7df514e739ed4b86923aecd42221ff60"
b_save_figures = True
fontsize = 22
b_correlations = False  # Whether to plot correlation functions
b_stabilizers = True  # Whether to plot three-qubit stabilizers

_1q_plot_components = ["x", "y", "z"]
_1q_plot_indices = [0, 1, 2]
_2q_plot_components = ["xx", "xz", "xy"]
_2q_plot_indices = [(0, 1), (0, 2)]

s_output_path = os.path.abspath("../output/") + "/"
s_plot_path = s_output_path + "analysis/"

files = find_db_files(s_output_path)
parameters = get_simulation_dict(s_output_path, s_unique_id)
s_data_path, _ = generate_paths(
    s_output_path, b_make_paths=False, s_data_subdir="simulations/"
)
s_data_path += S_FILE_PREFIX
s_output_file = s_data_path + "." + s_unique_id
result = LindbladMPOSolver.load_output(s_output_file)

if not os.path.exists(s_plot_path):
    os.mkdir(s_plot_path)

s_file_prefix = s_plot_path + s_unique_id
for s_obs_name in _1q_plot_components:
    plot_1q_obs_curves(
        parameters,
        result,
        s_obs_name,
        _1q_plot_indices,
        fontsize=fontsize,
        b_save_figures=b_save_figures,
        s_file_prefix=s_file_prefix,
    )
if b_correlations:
    for s_obs_name in _2q_plot_components:
        plot_2q_correlation_curves(
            parameters,
            result,
            s_obs_name,
            _2q_plot_indices,
            fontsize=fontsize,
            b_save_figures=b_save_figures,
            s_file_prefix=s_file_prefix,
        )

if b_stabilizers:
    plot_simulation_stabilizers(s_unique_id, fontsize)

plt.show()
tmp = 2  # Put a breakpoint here if desired
