# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from project_simulations.simulation_routines import *

# Set plot parameters
plt.rc("text", usetex=True)  # set usetex=False if tex isn't installed
plt.rc("font", family="serif")
plt.rcParams["font.size"] = 18
plt.rcParams["axes.linewidth"] = 2

# (a) initial |+,+,+> state, (b) initial 3Q chain graph state
s_simulation_id_a = "299f1297db9347e7853fb5eb510c08e5"
s_simulation_id_b = "d58e318d2b2c4cb0974ae402d01e3fba"

# The simulation also calculated the initialization of the graph state,
# Set b_plot_initial=True to plot it
b_plot_initial = False

# Generate paths
(
    s_output_path,
    s_evolution_path,
    s_estimation_path,
    s_simulation_path,
    _,
) = experiment_routines.generate_paths()
s_paper_plot_path = s_output_path + "paper_figures/"
if not os.path.exists(s_paper_plot_path):
    os.mkdir(s_paper_plot_path)

# Upper figure panel (a)
s_simulation_csv = s_output_path + S_DB_FILENAME
parameters = get_simulation_dict(s_simulation_csv, s_simulation_id_a)
s_output_file = s_simulation_path + S_FILE_PREFIX + "." + s_simulation_id_a
result_MPO = LindbladMPOSolver.load_output(s_output_file)

N = parameters["N"]
t_init = parameters["t_init"]
t_final = parameters["t_final"]
time_unit = parameters["time_unit"]
t_gates = parameters["t_gates"]
s_evolution_csv = s_output_path + experiment_routines.S_EVOLUTION_DF_FILENAME
s_evolution_id = parameters["evolution_id"]
evolution_dict = get_simulation_dict(s_evolution_csv, s_evolution_id)
s_pickle_file = (
    s_evolution_path
    + experiment_routines.S_EVOLUTION_DATA_PREFIX
    + "."
    + s_evolution_id
)
s_file_prefix = (
    s_paper_plot_path
    + experiment_routines.S_EVOLUTION_DATA_PREFIX
    + f".N={N}."
    + s_simulation_id_a
)
print(
    f'\nEvolution experiment {s_evolution_id}, backend: {evolution_dict["backend"]}. N = {N}.'
)
print(evolution_dict["experiment_link"])

sim_qubits = np.asarray(range(N))
promote_indices(sim_qubits)

# Load parameters for readout error mitigation
p_0_given_0, p_0_given_1 = load_mitigation_parameters(s_evolution_id)
topo_index = (
    experiment_routines.RING_TOPOLOGY_INDEXES
    if N == 12
    else experiment_routines.CHAIN_3Q_TOPOLOGY_INDEXES
)
xzz_list = (
    experiment_routines.RING_3Q_XZZ_LIST
    if N == 12
    else experiment_routines.CHAIN_3Q_XZZ_LIST
)
(
    times,
    qubits,
    ev_1Q_dict,
    correlations_dict,
    stabilizers_dict,
) = load_observables(
    s_pickle_file,
    topo_index,
    xzz_list,
    b_mitigate_readout=True,
    b_correlations=False,
    p_0_given_0=p_0_given_0,
    p_0_given_1=p_0_given_1,
    b_stabilizers=False,
)
times = times / time_unit

physical_qubits = evolution_dict["qubits"].strip("][").split(", ")

_1q_plot_indices = np.asarray(range(N))
n_plot_qubits = len(_1q_plot_indices)
axs = None
figs = []
axs_list = []
for i_plot_qubit in range(n_plot_qubits):
    qubit = qubits[_1q_plot_indices[i_plot_qubit]]
    physical_qubit = physical_qubits[_1q_plot_indices[i_plot_qubit]]
    sim_qubit = sim_qubits[_1q_plot_indices[i_plot_qubit]]
    RamX_data = unp_n(ev_1Q_dict[f"X_{qubit}"])
    RamX_err = unp_s(ev_1Q_dict[f"X_{qubit}"])
    RamY_data = unp_n(ev_1Q_dict[f"Y_{qubit}"])
    RamY_err = unp_s(ev_1Q_dict[f"Y_{qubit}"])

    s_obs_name = "x"
    obs_data_MPO_x, _ = prepare_curve_data(
        result_MPO, "obs-1q", s_obs_name, (sim_qubit,)
    )

    s_obs_name = "y"
    obs_data_MPO_y, _ = prepare_curve_data(
        result_MPO, "obs-1q", s_obs_name, (sim_qubit,)
    )

    fig, axs = plt.subplots(2, 1, figsize=(6, 6.5), sharex=True)
    ax = axs[0]

    ax.errorbar(
        times[::2],
        RamX_data[::2],
        yerr=RamX_err[::2],
        fmt="o",
        alpha=0.9,
        capsize=4,
        markersize=4,
        label=f"$\\left< X_{{{str(physical_qubit)}}} \\right>,$ exp",
        color="C0",
    )
    ax.errorbar(
        obs_data_MPO_x[0],
        obs_data_MPO_x[1],
        fmt="-",
        alpha=0.8,
        capsize=4,
        markersize=5,
        label=f"$\\left< X_{{{str(physical_qubit)}}} \\right>,$ sim",
        color="C0",
    )

    ax.errorbar(
        times[::2],
        RamY_data[::2],
        yerr=RamY_err[::2],
        fmt=">",
        alpha=0.9,
        capsize=4,
        markersize=4,
        label=f"$\\left< Y_{{{str(physical_qubit)}}} \\right>,$ exp",
        color="C1",
    )
    ax.errorbar(
        obs_data_MPO_y[0],
        obs_data_MPO_y[1],
        fmt="--",
        alpha=0.8,
        capsize=4,
        markersize=5,
        label=f"$\\left< Y_{{{str(physical_qubit)}}} \\right>,$ sim",
        color="C1",
    )
    # Edit the major and minor ticks of the x and y axes
    ax.xaxis.set_tick_params(which="major", size=7, width=2, direction="in", top="on")
    ax.xaxis.set_tick_params(which="minor", size=5, width=2, direction="in", top="on")
    ax.yaxis.set_tick_params(which="major", size=7, width=2, direction="in", right="on")
    ax.yaxis.set_tick_params(which="minor", size=5, width=2, direction="in", right="on")

    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=14)

    figs.append(fig)
    axs_list.append(axs)


# Lower figure panel (b)
s_simulation_csv = s_output_path + S_DB_FILENAME
parameters = get_simulation_dict(s_simulation_csv, s_simulation_id_b)
s_output_file = s_simulation_path + S_FILE_PREFIX + "." + s_simulation_id_b
result_MPO_stab = LindbladMPOSolver.load_output(s_output_file)

N = parameters["N"]
t_init = parameters["t_init"]
t_final = parameters["t_final"]
time_unit = parameters["time_unit"]
t_gates = parameters["t_gates"]
s_evolution_csv = s_output_path + experiment_routines.S_EVOLUTION_DF_FILENAME
s_evolution_id = parameters["evolution_id"]
evolution_dict = get_simulation_dict(s_evolution_csv, s_evolution_id)
s_pickle_file = (
    s_evolution_path
    + experiment_routines.S_EVOLUTION_DATA_PREFIX
    + "."
    + s_evolution_id
)
print(
    f'\nEvolution experiment {s_evolution_id}, backend: {evolution_dict["backend"]}. N = {N}.'
)
print(evolution_dict["experiment_link"])

sim_qubits = np.asarray(range(N))
promote_indices(sim_qubits)
p_0_given_0, p_0_given_1 = load_mitigation_parameters(s_evolution_id)
topo_index = (
    experiment_routines.RING_TOPOLOGY_INDEXES
    if N == 12
    else experiment_routines.CHAIN_3Q_TOPOLOGY_INDEXES
)
xzz_list = (
    experiment_routines.RING_3Q_XZZ_LIST
    if N == 12
    else experiment_routines.CHAIN_3Q_XZZ_LIST
)
(
    times,
    qubits,
    ev_1Q_dict,
    correlations_dict,
    stabilizers_dict,
) = load_observables(
    s_pickle_file,
    topo_index,
    xzz_list,
    b_mitigate_readout=True,
    b_correlations=False,
    p_0_given_0=p_0_given_0,
    p_0_given_1=p_0_given_1,
    b_stabilizers=True,
)
if b_plot_initial:
    times = times / time_unit + 1 * t_gates
else:
    times = times / time_unit

physical_qubits = evolution_dict["qubits"].strip("][").split(", ")

s_obs_name = "xzz"
_3q_plot_indices = (
    experiment_routines.RING_3Q_XZZ_LIST
    if N == 12
    else experiment_routines.CHAIN_3Q_XZZ_LIST
)
_3q_sim_plot_indices = _3q_plot_indices.copy()
promote_indices(_3q_sim_plot_indices)
_1q_plot_indices = (
    experiment_routines.RING_TOPOLOGY_INDEXES
    if N == 12
    else experiment_routines.CHAIN_3Q_TOPOLOGY_INDEXES
)
_1q_sim_plot_indices = _3q_plot_indices.copy()
promote_indices(_1q_sim_plot_indices)

s_file_prefix = (
    s_paper_plot_path
    + experiment_routines.S_EVOLUTION_DATA_PREFIX
    + f".N={N}."
    + s_simulation_id_b
)
stab = stabilizers_dict

_3q_tuple = _3q_plot_indices[0]
i, j, k = _3q_tuple[0], _3q_tuple[1], _3q_tuple[2]
q1, q2, q3 = physical_qubits[i], physical_qubits[j], physical_qubits[k]
i_x_qubit = 1
x1 = physical_qubits[i_x_qubit]

fig, ax = figs[1], axs_list[1][1]
stab_v = stab[f"stabilizer_{1}"]
XZZ = unp_n(stab_v)
XZZ_err = unp_s(stab_v)

obs_data_MPO_x, _ = prepare_curve_data(
    result_MPO_stab, "obs-1q", "x", (sim_qubits[i_x_qubit],)
)

if b_plot_initial:
    ax.errorbar(
        np.asarray(obs_data_MPO_x[0]),
        obs_data_MPO_x[1],
        fmt="-.",
        alpha=0.9,
        capsize=4,
        markersize=5,
        color="C0",
    )
else:
    gate_time_index = np.argmax(np.asarray(obs_data_MPO_x[0]) > t_gates)
    ax.errorbar(
        np.asarray(obs_data_MPO_x[0][gate_time_index:]) - t_gates,
        obs_data_MPO_x[1][gate_time_index:],
        fmt="-.",
        alpha=0.9,
        capsize=4,
        markersize=5,
        color="C0",
    )

s_x1 = f"$\\langle X_{{{q1}}}\\rangle$"

ax.errorbar(times, XZZ, XZZ_err, fmt="ro", alpha=0.8, capsize=4, markersize=5)

obs_data_MPO_stab, _ = prepare_curve_data(
    result_MPO_stab,
    "obs-3q",
    s_obs_name,
    (sim_qubits[i], sim_qubits[j], sim_qubits[k]),
)
if b_plot_initial:
    ax.errorbar(
        np.asarray(obs_data_MPO_stab[0]),
        obs_data_MPO_stab[1],
        fmt="-r",
        alpha=0.9,
        capsize=4,
        markersize=5,
    )
else:
    ax.errorbar(
        np.asarray(obs_data_MPO_stab[0][gate_time_index:]) - t_gates,
        obs_data_MPO_stab[1][gate_time_index:],
        fmt="-r",
        alpha=0.9,
        capsize=4,
        markersize=5,
    )

s_stab = f"$\\langle Z_{{{q2}}} X_{{{q1}}} Z_{{{q3}}}\\rangle$"

ax.xaxis.set_tick_params(which="major", size=7, width=2, direction="in", top="on")
ax.xaxis.set_tick_params(which="minor", size=5, width=2, direction="in", top="on")
ax.yaxis.set_tick_params(which="major", size=7, width=2, direction="in", right="on")
ax.yaxis.set_tick_params(which="minor", size=5, width=2, direction="in", right="on")
ax.set_xlabel(r"time [$\mu s$]", labelpad=1)
ax.legend(
    [s_x1 + ", sim", s_stab + ", exp", s_stab + ", sim"],
    loc="upper right",
    frameon=False,
    fontsize=14,
)
ax.set_xlim([-3, 100])
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.show()
