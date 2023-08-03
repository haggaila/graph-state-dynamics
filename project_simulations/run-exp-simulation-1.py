# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from project_simulations.simulation_routines import *

# This file is used to load the data of an experiment, run corresponding simulations (with
#    the estimated parameters), and plot the results.
# -----------------------------------------------------------------------------
# Plotting and simulation parameters
b_save_figures = True
b_save_to_db = True
fontsize = 22
b_mitigate_readout = True  # Whether to mitigate readout errors for the experiment plots
b_correlations = False  # Whether to simulate and plot correlation functions

# -----------------------------------------------------------------------------
# Simulation parameters to vary - choose the evolution experiment's id, and the solver type.
s_evolution_id = "9952b4b8a80744a798ee415970185d11"
solver = "mpo"  # 'mpo', 'scipy'

time_unit = 1e-6
tau = 0.05e-6 / time_unit

(
    topology,
    t_init,
    t_final,
    N,
    init_product_state,
    init_cz_gates,
    apply_gates,
    g_0,
    g_1,
    g_2,
    h_z,
    nu_p,
    b,
    J_z,
    qubits,
    t_gates,
    b_stabilizers,
    custom_observables,
) = load_experiment_parameters(s_evolution_id, time_unit)

# Create the metadata (database) dictionary, which defines our simulation. The default one provides
# empty values for all necessary fields
sim_metadata = DEF_METADATA.copy()
sim_metadata.update(
    {
        "N": N,
        "t_gates": t_gates,
        "solver": solver,
        "topology": topology,
        "qubits": str(qubits),
        "t_init": t_init,
        "t_final": t_final,
        "tau": tau,
        "time_unit": time_unit,
        "evolution_id": s_evolution_id,
    }
)

# Now define the solver-specific parameters, and update the metadata
if solver == "mpo":
    min_dim_rho = 0
    max_dim_rho = 100
    cut_off_rho = 1e-19
    force_rho_Hermitian_step = 3

    sim_metadata.update(
        {
            "cut_off_rho": cut_off_rho,
            "max_dim_rho": max_dim_rho,
            "force_rho_Hermitian_step": force_rho_Hermitian_step,
        }
    )
elif solver == "scipy":
    method = "RK45"  # 'RK45', 'DOP853'
    atol = 1e-8
    rtol = 1e-8
    sim_metadata.update({"method": method, "atol": atol, "rtol": rtol})

# Use the metadata dictionary to build parameters dictionary, create a solver, solve,
# save the results, plot simulation figures, and save database entries.
solve_simulation(
    sim_metadata,
    init_product_state,
    init_cz_gates,
    apply_gates,
    g_0,
    g_1,
    g_2,
    h_z,
    nu_p,
    b,
    J_z,
    fontsize=fontsize,
    b_save_to_db=b_save_to_db,
    b_save_figures=b_save_figures,
    b_stabilizers=b_stabilizers,
    custom_observables=custom_observables,
)

plot_experiment_simulation(
    sim_metadata["unique_id"],
    fontsize=fontsize,
    b_mitigate_readout=b_mitigate_readout,
    b_correlations=b_correlations,
    b_stabilizers=b_stabilizers,
    b_save_figures=b_save_figures,
    b_plot_z=False,
)

plt.show()
tmp = 2  # Put a breakpoint here if desired
