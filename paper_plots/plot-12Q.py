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

# (a) 12Q graph state without DD, (b) 12Q graph state with DD
s_simulation_id_a = "b2cdcb62bb894c02981b21c191c852fd"
s_simulation_id_b = "13e7cea845524fec84bec865d20e4f75"

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

# Upper figure panel (a) / idle
s_simulation_csv = s_output_path + S_DB_FILENAME
parameters = get_simulation_dict(s_simulation_csv, s_simulation_id_a)
s_output_file = s_simulation_path + S_FILE_PREFIX + "." + s_simulation_id_a
result = LindbladMPOSolver.load_output(s_output_file)

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
    + s_simulation_id_a
)
stab = stabilizers_dict

# Initial value for summing the stabilizers
stab_norm_MPO = 0
stab_norm = 0

fidelity_a = result["obs-cu"].get(("ideal_proj", ()))

fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 7))
axs = axs.reshape(12)
count = 0

for i_x_qubit, _3q_tuple in zip(_1q_plot_indices, _3q_plot_indices):
    i, j, k = _3q_tuple[0], _3q_tuple[1], _3q_tuple[2]
    q1, q2, q3 = physical_qubits[i], physical_qubits[j], physical_qubits[k]
    x1 = physical_qubits[i_x_qubit]
    ax = axs[count]
    stab_v = stab[f"stabilizer_{i}"]
    XZZ = unp_n(stab_v)
    XZZ_err = unp_s(stab_v)
    ax.errorbar(times, XZZ, XZZ_err, fmt="ro", alpha=0.8, capsize=4, markersize=5)

    obs_data_MPO_stab, _ = prepare_curve_data(
        result, "obs-3q", s_obs_name, (sim_qubits[i], sim_qubits[j], sim_qubits[k])
    )
    stab_norm_MPO += (np.asarray(obs_data_MPO_stab[1]) + 1) / 2
    stab_norm += (stab_v + 1) / 2

    if b_plot_initial:
        ax.errorbar(
            np.asarray(obs_data_MPO_stab[0]),
            obs_data_MPO_stab[1],
            fmt="-r",
            alpha=1,
            capsize=4,
            markersize=5,
        )
    else:
        gate_time_index = np.argmax(np.asarray(obs_data_MPO_stab[0]) > t_gates)
        ax.errorbar(
            np.asarray(obs_data_MPO_stab[0])[gate_time_index:] - t_gates,
            obs_data_MPO_stab[1][gate_time_index:],
            fmt="-r",
            alpha=1,
            capsize=4,
            markersize=5,
        )

    s_stab = f"$\\langle Z_{{{q2}}} X_{{{q1}}} Z_{{{q3}}}\\rangle$"

    ax.legend(
        [s_stab + ", exp", s_stab + ", sim"],
        loc="upper right",
        frameon=False,
        fontsize=12,
    )
    ax.xaxis.set_tick_params(which="major", size=7, width=2, direction="in", top="on")
    ax.xaxis.set_tick_params(which="minor", size=5, width=2, direction="in", top="on")
    ax.yaxis.set_tick_params(which="major", size=7, width=2, direction="in", right="on")
    ax.yaxis.set_tick_params(which="minor", size=5, width=2, direction="in", right="on")
    count += 1

fig.supxlabel(r"time [$\mu s$]", y=0.05)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05, wspace=0.03)

# Upper figure panel (a) /  ZZ DD
s_simulation_csv = s_output_path + S_DB_FILENAME
parameters = get_simulation_dict(s_simulation_csv, s_simulation_id_b)
s_output_file = s_simulation_path + S_FILE_PREFIX + "." + s_simulation_id_b
result = LindbladMPOSolver.load_output(s_output_file)

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
    times_DD,
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
    times_DD = times_DD / time_unit + 1 * t_gates
else:
    times_DD = times_DD / time_unit

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
colors = cm.get_cmap("cool", 10)


stab_norm_MPO_DD = 0
stab_norm_DD = 0
fidelity_b = result["obs-cu"].get(("ideal_proj", ()))
fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 7))
axs = axs.reshape(12)
count = 0
for i_x_qubit, _3q_tuple in zip(_1q_plot_indices, _3q_plot_indices):
    i, j, k = _3q_tuple[0], _3q_tuple[1], _3q_tuple[2]
    q1, q2, q3 = physical_qubits[i], physical_qubits[j], physical_qubits[k]
    x1 = physical_qubits[i_x_qubit]
    stab_v = stab[f"stabilizer_{i}"]
    XZZ = unp_n(stab_v)
    XZZ_err = unp_s(stab_v)

    ax = axs[count]
    ax.errorbar(times_DD, XZZ, XZZ_err, fmt="co", alpha=0.8, capsize=4, markersize=5)

    obs_data_MPO_stab_DD, _ = prepare_curve_data(
        result, "obs-3q", s_obs_name, (sim_qubits[i], sim_qubits[j], sim_qubits[k])
    )
    stab_norm_MPO_DD += (np.asarray(obs_data_MPO_stab_DD[1]) + 1) / 2
    stab_norm_DD += (stab_v + 1) / 2

    if b_plot_initial:
        ax.errorbar(
            np.asarray(obs_data_MPO_stab_DD[0]),
            obs_data_MPO_stab_DD[1],
            fmt="-c",
            alpha=1,
            capsize=4,
            markersize=5,
        )
    else:
        ax.errorbar(
            np.asarray(obs_data_MPO_stab_DD[0])[gate_time_index:] - t_gates,
            obs_data_MPO_stab_DD[1][gate_time_index:],
            fmt="-c",
            alpha=1,
            capsize=4,
            markersize=5,
        )

    s_stab = f"$\\langle Z_{{{q2}}} X_{{{q1}}} Z_{{{q3}}}\\rangle$"

    count += 1
    ax.legend(
        [s_stab + ", exp", s_stab + ", sim"],
        loc="lower right",
        frameon=False,
        fontsize=12,
    )
    ax.xaxis.set_tick_params(which="major", size=7, width=2, direction="in", top="on")
    ax.xaxis.set_tick_params(which="minor", size=5, width=2, direction="in", top="on")
    ax.yaxis.set_tick_params(which="major", size=7, width=2, direction="in", right="on")
    ax.yaxis.set_tick_params(which="minor", size=5, width=2, direction="in", right="on")
fig.supxlabel(r"time [$\mu s$]", y=0.05)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05, wspace=0.03)


fig, axs = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
ax = axs[0]
ax.errorbar(
    times_DD,
    unp_n(stab_norm_DD / 12),
    yerr=unp_s(stab_norm_DD / 12),
    fmt=">r",
    alpha=0.9,
    capsize=4,
    markersize=5,
    label=r"ZZ DD, exp",
)
if b_plot_initial:
    ax.errorbar(
        np.asarray(obs_data_MPO_stab[0]),
        stab_norm_MPO_DD / 12,
        fmt="-r",
        alpha=0.9,
        capsize=4,
        markersize=5,
        linewidth=2,
        label=r"ZZ DD, sym",
    )
else:
    ax.errorbar(
        np.asarray(obs_data_MPO_stab[0][gate_time_index:]) - t_gates,
        stab_norm_MPO_DD[gate_time_index:] / 12,
        fmt="-r",
        alpha=0.9,
        capsize=4,
        markersize=5,
        linewidth=2,
        label=r"ZZ DD, sym",
    )


ax.errorbar(
    times,
    unp_n(stab_norm / 12),
    yerr=unp_s(stab_norm / 12),
    fmt="oc",
    alpha=0.9,
    capsize=4,
    markersize=5,
    label=r"Idle, exp",
)
if b_plot_initial:
    ax.errorbar(
        np.asarray(obs_data_MPO_stab[0]),
        stab_norm_MPO / 12,
        fmt="--c",
        alpha=0.9,
        capsize=4,
        markersize=5,
        linewidth=2,
        label=r"Idle, sym",
    )
else:
    ax.errorbar(
        np.asarray(obs_data_MPO_stab[0][gate_time_index:]) - t_gates,
        stab_norm_MPO[gate_time_index:] / 12,
        fmt="--c",
        alpha=0.9,
        capsize=4,
        markersize=5,
        linewidth=2,
        label=r"Idle, sym",
    )


ax.xaxis.set_tick_params(which="major", size=7, width=2, direction="in", top="on")
ax.xaxis.set_tick_params(which="minor", size=3, width=2, direction="in", top="on")
ax.yaxis.set_tick_params(which="major", size=7, width=2, direction="in", right="on")
ax.yaxis.set_tick_params(which="minor", size=3, width=2, direction="in", right="on")
ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=18)
ax.set_ylabel(
    "$\\frac{1}{12}\sum_i \\frac{1 + \\left< S_i \\right>}{2}$", labelpad=5, fontsize=24
)

ax = axs[1]

if b_plot_initial:
    ax.errorbar(
        np.asarray(fidelity_b[0]),
        np.asarray(fidelity_b[1]) * 2**12,
        fmt="-r",
        alpha=0.9,
        capsize=4,
        linewidth=2,
        markersize=5,
        label=r"ZZ DD, sym",
    )

    ax.errorbar(
        np.asarray(fidelity_a[0]),
        np.asarray(fidelity_a[1]) * 2**12,
        fmt="--c",
        linewidth=2,
        alpha=0.9,
        capsize=4,
        markersize=5,
        label=r"Idle, sym",
    )
else:
    ax.errorbar(
        np.asarray(fidelity_b[0][gate_time_index:]) - t_gates,
        np.asarray(fidelity_b[1][gate_time_index:]) * 2**12,
        fmt="-r",
        alpha=0.9,
        capsize=4,
        linewidth=2,
        markersize=5,
        label=r"ZZ DD, sym",
    )

    ax.errorbar(
        np.asarray(fidelity_a[0][gate_time_index:]) - t_gates,
        np.asarray(fidelity_a[1][gate_time_index:]) * 2**12,
        fmt="--c",
        linewidth=2,
        alpha=0.9,
        capsize=4,
        markersize=5,
        label=r"Idle, sym",
    )

ax.set_xlabel(r"time [$\mu s$]", labelpad=1, fontsize=22)
ax.set_yscale("log")
ax.xaxis.set_tick_params(which="major", size=7, width=2, direction="in", top="on")
ax.xaxis.set_tick_params(which="minor", size=3, width=2, direction="in", top="on")
ax.yaxis.set_tick_params(which="major", size=7, width=2, direction="in", right="on")
ax.yaxis.set_tick_params(which="minor", size=3, width=2, direction="in", right="on")
ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=18)
ax.set_ylabel("Fidelity", labelpad=5, fontsize=24)
plt.tight_layout()
plt.subplots_adjust(hspace=0.05, wspace=0.03)
plt.show()
