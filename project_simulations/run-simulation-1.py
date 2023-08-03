# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from time import sleep

from project_simulations.simulation_routines import *

# This file is used to run the dynamical simulations of the project without corresponding
#    experiments (by specifying the system parameters explicitly), and plot the results.
# -----------------------------------------------------------------------------
# Plotting and simulation parameters
b_save_figures = False
b_save_to_db = True
fontsize = 22
b_correlations = False  # Whether to simulate and plot correlation functions
b_stabilizers = True  # Whether to simulate and plot three-qubit stabilizers

# -----------------------------------------------------------------------------
# System parameters to vary
N = 3
topology = "chain.E"
solver = "mpo"  # 'mpo', 'scipy'
kHz = 2 * np.pi * 1e-3

T_1 = [63.0, 66.0, 76.0]
T_2 = [89.0, 62.0, 70.0]
g_0 = 1.0 * np.asarray(T_1) ** -1
g_1 = 0.0 * np.ones(N)
g_2 = 1.0 * (np.asarray(T_2) ** -1 - 0.5 * g_0) / 2
h_z = 1.0 * np.asarray([-21 * kHz, -(-17) * kHz, -25 * kHz])
nu_p = 1.0 * np.asarray([17 * kHz, 26 * kHz, 21 * kHz])
b = 0.5 * np.ones(N)
apply_gates = []
init_cz_gates = [(0, 1), (1, 2)]
init_product_state = ["+x"] * N
# init_product_state[0] = (0.5**0.5, 0.05)

J_z = np.zeros((N, N))
J_z[0, 1] = 26 * kHz
J_z[1, 2] = 21 * kHz
J_z[0, 2] = 29 * kHz
J = np.zeros((N, N))
J[0, 2] = 0 * kHz

t_init = 0.0
t_final = 10
load_unique_id = ""
tau = 0.1

# Create the metadata (database) dictionary, which defines our simulation. The default one provides
# empty values for all necessary fields
sim_metadata = DEF_METADATA.copy()
qubits = [i for i in range(N)]
sim_metadata.update(
    {
        "N": N,
        "solver": solver,
        "topology": topology,
        "qubits": str(qubits),
        "t_init": t_init,
        "t_final": t_final,
        "tau": tau,
        "load_unique_id": load_unique_id,
    }
)

# Now define the solver-specific parameters, and update the metadata
if solver == "mpo":
    min_dim_rho = 0
    max_dim_rho = 60
    cut_off_rho = 1e-18
    force_rho_Hermitian_step = 5

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
    J,
    fontsize=fontsize,
    b_save_to_db=b_save_to_db,
    b_save_figures=b_save_figures,
    b_stabilizers=b_stabilizers,
)

sleep(3)
if b_stabilizers:
    plot_simulation_stabilizers(sim_metadata["unique_id"], fontsize)

plt.show()
tmp = 2  # Put a breakpoint here if desired
