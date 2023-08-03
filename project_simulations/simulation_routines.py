# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Routines for managing the research project running multiple simulations of experiments.
"""

import os.path
import json
from typing import Container

from mpl_toolkits.mplot3d import Axes3D

from lindbladmpo.examples.qubit_driving import topologies
from lindbladmpo.examples.qubit_driving.output_routines import *
from lindbladmpo.plot_routines import *
from lindbladmpo.LindbladMPOSolver import *
from project_experiments import experiment_routines
from project_experiments.error_mitigation import *
import matplotlib as mpl
from matplotlib import cm


DEF_METADATA = {
    "unique_id": "",
    "evolution_id": "",
    "N": "",
    "solver": "",
    "t_init": "",
    "t_final": "",
    "tau": "",
    "time_unit": "",
    "max_dim_rho": "",
    "cut_off_rho": "",
    "method": "",
    "atol": "",
    "rtol": "",
    "load_unique_id": "",
    "force_rho_Hermitian_step": "",
    "topology": "",
    "qubits": "",
    "t_gates": "",
}
"""Default metadata and db parameters for the simulations in this project."""

S_DB_FILENAME = "simulations.csv"
"""File name to use for the database of the project simulations."""

S_FILE_PREFIX = "parity"
"""Prefix for the file names of all solver and plotting output files of the project simulations."""

unp_n = unp.nominal_values
unp_s = unp.std_devs
# Define shortcuts for the unumpy package


def _extend_parity_qubits(
    solver_params: Dict,
    parity_nu: Sequence,
    parity_b: Optional[Sequence] = None,
    b_parity_qubits_last=False,
):
    """
    Extends the parameters of a simulation that includes parity-oscillation with the required
        fictitious parity-oscillation qubits.
    Args:
        solver_params: Parameters passed to the solver.
        parity_nu: An array with the parity-oscillations frequency of each qubit.
        parity_b: An optional value for each fictitious qubit's density matrix population,
            if different from a fully mixed state.
        b_parity_qubits_last: An implementation detail, determining whether the parity qubits
            should be put at the end of the system, or rather each adjacent to its physical qubit.
    """

    N = solver_params["N"]

    init_pauli_state = solver_params.get("init_pauli_state", None)
    if init_pauli_state is not None:
        raise Exception("init_pauli_state is deprecated and should not be used.")
    init_product_state = solver_params.get("init_product_state", None)
    init_product_state = _extend_1d(init_product_state, N, "+x", object)
    r_qubits = range(N)
    if parity_b is None:
        parity_b = [0.5] * N
    for i in r_qubits:
        if b_parity_qubits_last:
            init_product_state[N + i] = parity_b[i]
        else:
            init_product_state[2 * i + 1] = parity_b[i]
    solver_params["init_product_state"] = init_product_state
    init_graph_state = solver_params.get("init_graph_state", None)
    promote_indices(init_graph_state)
    solver_params["init_graph_state"] = init_graph_state
    init_cz_gates = solver_params.get("init_cz_gates", None)
    promote_indices(init_cz_gates)
    solver_params["init_cz_gates"] = init_cz_gates
    apply_gates = solver_params.get("apply_gates", None)
    promote_indices(apply_gates)
    solver_params["apply_gates"] = apply_gates

    h_x = solver_params.get("h_x", None)
    if h_x is not None:
        h_x = _extend_1d(h_x, N)
    solver_params["h_x"] = h_x
    h_y = solver_params.get("h_y", None)
    if h_y is not None:
        h_y = _extend_1d(h_y, N)
    solver_params["h_y"] = h_y
    h_z = solver_params.get("h_z", None)
    if h_z is not None:
        h_z = _extend_1d(h_z, N)
    solver_params["h_z"] = h_z

    g_0 = solver_params.get("g_0", None)
    if g_0 is not None:
        g_0 = _extend_1d(g_0, N)
    solver_params["g_0"] = g_0
    g_1 = solver_params.get("g_1", None)
    if g_1 is not None:
        g_1 = _extend_1d(g_1, N)
    solver_params["g_1"] = g_1
    g_2 = solver_params.get("g_2", None)
    if g_2 is not None:
        g_2 = _extend_1d(g_2, N)
    solver_params["g_2"] = g_2

    J = solver_params.get("J", None)
    if J is not None:
        J = _extend_2d(J, N)
    solver_params["J"] = J
    J_z = solver_params.get("J_z", None)
    J_z = _extend_2d(J_z, N)
    for i in r_qubits:
        if b_parity_qubits_last:
            J_z[i, N + i] = J_z[i, N + i] - parity_nu[i]
        else:
            J_z[2 * i, 2 * i + 1] = J_z[2 * i, 2 * i + 1] - parity_nu[i]
        # A factor of 1/2 is included in the definition of J_z in the Hamiltonian
    solver_params["J_z"] = J_z
    a_indices = ["1q_indices", "2q_indices", "3q_indices"]
    for s_key in a_indices:
        indices = solver_params.get(s_key, None)
        if indices is not None:
            promote_indices(indices)
            solver_params[s_key] = indices

    solver_params["N"] = N * 2


def _extend_1d(val_arr, N, def_val: Any = 0.0, dtype=None, b_parity_qubits_last=False):
    """
    A helper function extending the parameters of a simulation that includes parity-oscillation
        with the required fictitious parity-oscillation qubits.
    """
    if val_arr is not None and not isinstance(val_arr, Container):
        val_arr = [val_arr] * N
    a = np.full(2 * N, def_val, dtype=dtype)
    if val_arr is not None:
        for i, el in enumerate(val_arr):
            if b_parity_qubits_last:
                a[i] = el
            else:
                a[2 * i] = el
    return a


def _extend_2d(val_arr, N, def_val=0.0, dtype=None, b_parity_qubits_last=False):
    """
    A helper function extending the parameters of a simulation that includes parity-oscillation
        with the required fictitious parity-oscillation qubits.
    """
    if val_arr is not None and not isinstance(val_arr, Container):
        val_arr = np.full((N, N), val_arr)
        # create a matrix of identical values
    a = np.full((2 * N, 2 * N), def_val, dtype=dtype)
    if val_arr is not None:
        r_qubits = range(N)
        for i in r_qubits:
            for j in r_qubits:
                if b_parity_qubits_last:
                    a[i, j] = val_arr[i, j]
                else:
                    a[2 * i, 2 * j] = val_arr[i, j]
    return a


def promote_indices(arr, b_parity_qubits_last=False):
    """
    A helper function that advances qubit indices in a simulation that includes parity-oscillation.
    """
    if arr is not None and not b_parity_qubits_last:
        for i_entry, q_entry in enumerate(arr):
            if isinstance(q_entry, tuple):
                if len(q_entry) == 2:  # It's a pair of two indices
                    arr[i_entry] = (2 * q_entry[0], 2 * q_entry[1])
                elif len(q_entry) == 3:
                    if isinstance(
                        q_entry[1], str
                    ):  # A 1Q tuple of 'apply_gates': t, gate, Q0
                        arr[i_entry] = (q_entry[0], q_entry[1], 2 * q_entry[2])
                    else:  # It's a triplet of indices
                        arr[i_entry] = (2 * q_entry[0], 2 * q_entry[1], 2 * q_entry[2])
                elif len(q_entry) == 4:  # A 2Q tuple of 'apply_gates': t, gate, Q0, Q1
                    arr[i_entry] = (
                        q_entry[0],
                        q_entry[1],
                        2 * q_entry[2],
                        2 * q_entry[3],
                    )
            else:
                arr[i_entry] = 2 * q_entry


def promote_custom_obs_indices(arr, b_parity_qubits_last=False):
    """
    A helper function that advances qubit indices in a simulation that includes parity-oscillation.
    """
    if arr is not None and not b_parity_qubits_last:
        for i_entry, q_entry in enumerate(arr):
            if isinstance(q_entry, tuple):
                if len(q_entry) == 2:  # Second entry is a qubit index
                    arr[i_entry] = (q_entry[0], 2 * q_entry[1])
                elif len(q_entry) == 3:
                    arr[i_entry] = (q_entry[0], 2 * q_entry[1], 2 * q_entry[2])


def plot_device(
    N: int,
    coupling_map: List[List[int]],
    qubit_coord: List[List[int]],
    b_save_figures: bool,
    s_file_prefix: str,
    b_transpose_plot=False,
):
    """Plots a schematic drawing of a device with its qubits and their coupling."""
    qubit_color = ["#648fff"] * N
    figsize = (7, 4)
    q_coord = []
    if b_transpose_plot:
        figsize = (figsize[1], figsize[0])
        for ll in qubit_coord:
            q_coord.append([ll[1], ll[0]])
    else:
        q_coord = qubit_coord

    try:
        from qiskit.visualization.gate_map import plot_coupling_map

        _ = plot_coupling_map(
            num_qubits=N,
            qubit_coordinates=q_coord,
            coupling_map=coupling_map,
            figsize=figsize,
            qubit_color=qubit_color,
        )
        if b_save_figures:
            plt.savefig(s_file_prefix + ".png")
        plt.draw()
        plt.pause(0.1)
        plt.show(block=False)
    except Exception as e:
        print(str(e))


def load_experiment_parameters(s_evolution_id: str, time_unit: float):
    """Loads parameters of an experiment and returns simulation parameters"""
    (
        s_output_path,
        s_evolution_path,
        s_estimation_path,
        _,
        _,
    ) = experiment_routines.generate_paths()
    s_evolution_csv = s_output_path + experiment_routines.S_EVOLUTION_DF_FILENAME
    evolution_dict = get_simulation_dict(s_evolution_csv, s_evolution_id)
    qubits = json.loads(evolution_dict["qubits"])
    N = len(qubits)
    r_qubits = range(N)
    J_z = np.zeros((N, N))
    t_init = 0.0
    t_final = evolution_dict["t_final"] / time_unit
    init_state = evolution_dict["init_state"]
    offset_freq = evolution_dict["offset_freq"]
    Hz = 2 * np.pi * time_unit

    s_db_path = (
        s_estimation_path
        + experiment_routines.S_ESTIMATION_DATA_PREFIX
        + "."
        + evolution_dict["estimation_id"]
        + ".csv"
    )
    df = pd.read_csv(s_db_path, parse_dates=None)

    s_topology = evolution_dict["topology"]
    device_map = topologies.coupling_maps[s_topology]
    coupling_map = [
        (pair[0], pair[1])
        for pair in device_map
        if pair[0] in qubits and pair[1] in qubits
    ]
    for pair in coupling_map:
        inverse_pair = (pair[1], pair[0])
        if inverse_pair in coupling_map:
            coupling_map.remove(inverse_pair)
    sim_coupling_map = [
        (qubits.index(pair[0]), qubits.index(pair[1])) for pair in coupling_map
    ]
    init_product_state: Any = ["+x"] * N
    # init_cz_gates = sim_coupling_map.copy() if init_state == "gr" else []
    init_cz_gates = []
    apply_gates = []
    custom_obs = []
    custom_observables = []
    t = 0.0
    if init_state == "gr":
        for i in r_qubits:
            custom_obs.append(("h", i))  # Creates the initial |+> state
        s_cz_groups = evolution_dict["cz_groups"]
        s_gate_len_groups = evolution_dict["gate_len_groups"]
        if s_cz_groups is not None:
            cz_groups = json.loads(s_cz_groups)
            gate_lens = json.loads(s_gate_len_groups)
            for cz_group, len_group in zip(cz_groups, gate_lens):
                gate_len = np.max(len_group) / time_unit
                timing_fraction = 0.25  # MUST remain 0.25 for the code below!
                t += gate_len * timing_fraction
                for q_pair in cz_group:
                    apply_gates.append((t, "x", qubits.index(q_pair[0])))
                t += gate_len * timing_fraction
                for q_pair in cz_group:
                    apply_gates.append((t, "x", qubits.index(q_pair[1])))
                t += gate_len * timing_fraction
                for q_pair in cz_group:
                    apply_gates.append((t, "x", qubits.index(q_pair[0])))
                t += gate_len * timing_fraction
                for q_pair in cz_group:
                    apply_gates.append((t, "x", qubits.index(q_pair[1])))
                for q_pair in cz_group:
                    q_i = qubits.index(q_pair[0])
                    q_j = qubits.index(q_pair[1])
                    apply_gates.append((t, "cz", q_i, q_j))
                    custom_obs.append(("cz", q_i, q_j))
        custom_observables.append((("ideal_proj", "g"), custom_obs))
    t_final += t
    t_gates = t

    n_dd_cycles = evolution_dict["n_dd_cycles"]
    b_zz_dd = evolution_dict["b_zz_dd"]
    s_zz_dd_group_indexes = evolution_dict["zz_dd_group_indexes"]
    zz_dd_group_indexes = json.loads(s_zz_dd_group_indexes)
    if b_zz_dd:
        n_delays_per_cycle = 4
    else:
        n_delays_per_cycle = 2
    if n_dd_cycles:
        r_cycles = range(n_delays_per_cycle * int(n_dd_cycles))
        delays = json.loads(evolution_dict["delays"])
        t = t_gates
        for i_delay, delay in enumerate(delays):
            if i_delay == 0:
                if delay != 0.0:
                    raise Exception("When doing CPMG, the first delay must 0!")
            else:
                for i_cycle in r_cycles:
                    i_cycle_mod = np.mod(i_cycle, 2)
                    t += (
                        (delay - delays[i_delay - 1])
                        / (n_delays_per_cycle * n_dd_cycles)
                    ) / time_unit
                    for i_qubit, i_group in enumerate(zz_dd_group_indexes):
                        if i_group == i_cycle_mod:
                            apply_gates.append((t, "x", i_qubit))

    b_stabilizers = evolution_dict.get("b_stabilizers", False)

    zz_sign = -1.0
    parameters_1Q = ["T1", "T2_PO", "Delta_PO", "nu_PO", "x_0", "y_0", "z_0"]
    parameters_2Q = ["zz"]
    parameters_keys = {
        "T1": "T1",
        "T2_PO": "T2",
        "Delta_PO": "Delta",
        "nu_PO": "nu",
        "x_0": "x_0",
        "y_0": "y_0",
        "z_0": "z_0",
    }
    parameters_values = {
        "T1": [],
        "T2": [],
        "Delta": [],
        "nu": [],
        "x_0": [],
        "y_0": [],
        "z_0": [],
    }
    for par in parameters_1Q:
        df2 = df.query(f"variable_name == '{par}'")
        param_key = parameters_keys[par]
        for q in qubits:
            parameters_values[param_key].append(
                df2.query(f"device_components == 'Q{q}'")["value"].mean()
            )

    T_1 = parameters_values["T1"]
    T_2 = parameters_values["T2"]
    g_0 = (np.asarray(T_1) ** -1) * time_unit
    g_1 = 0.0 * np.ones(N) * time_unit
    g_2 = ((np.asarray(T_2) ** -1) * time_unit - 0.5 * g_0) / 2
    h_z = -(np.asarray(parameters_values["Delta"]) * Hz + offset_freq * Hz)
    x_0s = parameters_values.get("x_0", [])
    y_0s = parameters_values.get("y_0", [])
    z_0s = parameters_values.get("z_0", [])
    for i_qubit, x_0 in enumerate(x_0s):
        if np.isnan(x_0):
            x_0 = 0.0
        y_0 = y_0s[i_qubit] if not np.isnan(y_0s[i_qubit]) else 0.0
        z_0 = z_0s[i_qubit]
        x, y, z = (z_0, y_0, -x_0)  # The initial rotation to the +x state
        y = -y  # To compare with the experiment?
        init_product_state[i_qubit] = (0.5 * (1 + z), 0.5 * x, -0.5 * y)
    nu_p = np.asarray(parameters_values["nu"]) * Hz
    b = 0.5 * np.ones(N)

    for par in parameters_2Q:
        df2 = df.query(f"variable_name == '{par}'")
        for pair, sim_pair in zip(coupling_map, sim_coupling_map):
            zz_val = (
                zz_sign
                * df2.query(
                    f"device_components in ['Q{pair[0]},Q{pair[1]}',"
                    f"'Q{pair[1]},Q{pair[0]}']"
                )["value"].mean()
            )
            i0, i1 = sim_pair[0], sim_pair[1]
            J_z[i0, i1] = zz_val * Hz
            h_z[i0] = h_z[i0] - zz_val * Hz
            h_z[i1] = h_z[i1] - zz_val * Hz

    return (
        s_topology,
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
    )


def load_mitigation_parameters(s_evolution_id: str):
    """Loads parameters of an experiment and returns readout error mitigation parameters"""
    (
        s_output_path,
        s_evolution_path,
        s_estimation_path,
        _,
        _,
    ) = experiment_routines.generate_paths()
    s_evolution_csv = s_output_path + experiment_routines.S_EVOLUTION_DF_FILENAME
    evolution_dict = get_simulation_dict(s_evolution_csv, s_evolution_id)
    qubits = json.loads(evolution_dict["qubits"])

    s_db_path = (
        s_estimation_path
        + experiment_routines.S_ESTIMATION_DATA_PREFIX
        + "."
        + evolution_dict["estimation_id"]
        + ".csv"
    )
    df = pd.read_csv(s_db_path, parse_dates=None)

    # parameters_1Q = ["A_PO", "B_PO"]
    # parameters_keys = {"A_PO": "A", "B_PO": "B"}
    parameters_1Q = ["pi_z", "pi_0"]
    parameters_keys = {"pi_z": "pi_z", "pi_0": "pi_0"}
    parameters_values = {"A": [], "B": [], "pi_z": [], "pi_0": []}
    for par in parameters_1Q:
        filtered_df = df.query(f"variable_name=='{par}'")[
            ["value", "std_dev", "device_components"]
        ]
        param_key = parameters_keys[par]
        for q in qubits:
            q_df = filtered_df.query(f"device_components=='Q{q}'")
            val = q_df["value"].values[0]
            err = q_df["std_dev"].values[0]
            parameters_values[param_key].append(ufloat(val, err))
    p_0_given_0, p_0_given_1 = None, None
    # A = np.asarray(parameters_values.get("A", []))
    # B = np.asarray(parameters_values.get("B", []))
    # if len(A) > 0 and len(B):
    #     p_0_given_0 = A + B
    #     p_0_given_1 = np.asarray(
    #         [(b - a) if b.n - a.n > 0.0 else ufloat(0.0, 0.0) for a, b in zip(A, B)]
    #     )
    pi_0 = np.asarray(parameters_values.get("pi_0", []))
    pi_z = np.asarray(parameters_values.get("pi_z", []))
    if len(pi_0) > 0 and len(pi_z):
        p_0_given_0 = pi_0 + pi_z
        p_0_given_1 = np.asarray(
            [
                (b - a) if b.n - a.n > 0.0 else ufloat(0.0, 0.0)
                for b, a in zip(pi_0, pi_z)
            ]
        )

    return p_0_given_0, p_0_given_1


def solve_simulation(
    sim_metadata: Dict,
    init_product_state,
    init_cz_gates,
    apply_gates,
    g_0,
    g_1,
    g_2,
    h_z,
    parity_nu,
    parity_b,
    J_z,
    J=None,
    fontsize=20,
    b_save_to_db=True,
    b_save_figures=True,
    b_stabilizers=False,
    custom_observables=None,
):
    """Solves and saves a dynamical simulation and plots some analysis figures."""
    N = sim_metadata["N"]
    s_solver = sim_metadata["solver"]
    t_init = sim_metadata["t_init"]
    t_final = sim_metadata["t_final"]
    tau = sim_metadata["tau"]
    load_unique_id = sim_metadata["load_unique_id"]
    force_rho_Hermitian_step = sim_metadata["force_rho_Hermitian_step"]

    # -------------------------------------------------------------------------
    # The following parameters are fixed for the project and kept hard-coded.
    _1q_components = ["x", "y", "z"]
    _2q_components = []  # ["xx", "yy"]  # ["xx", "yy", "zz", "xy", "xz", "yz"]
    _1q_plot_components = []  # ["x", "z"]
    _2q_plot_components = []  # ["xx", "yy"]
    _1q_indices = []
    _2q_indices = []

    if N < 2:
        raise Exception("At least 2 qubits must be simulated.")
    r_qubits = range(N)
    for i in r_qubits:
        _1q_indices.append(i)  # Request 1Q observables on all qubits
    if len(_2q_components):
        # Add 2Q observables for all qubits. Can be optimized.
        for i in r_qubits:
            for j in r_qubits:
                if i != j:
                    _2q_indices.append((i, j))
    if b_stabilizers:
        pass

    _1q_plot_indices = [0, 1]
    if N >= 3:
        _1q_plot_indices.append(2)
    if N >= 6:
        _1q_plot_indices.append((N - 3))
    if N >= 5:
        _1q_plot_indices.append((N - 2))
    if N >= 4:
        _1q_plot_indices.append((N - 1))
    _2q_plot_indices = [(0, 1)]
    if N >= 4:
        _2q_plot_indices.append((0, 2))
    if N >= 6:
        _2q_plot_indices.append((0, 3))
    if N >= 7:
        _2q_plot_indices.append((0, N - 3))
    if N >= 5:
        _2q_plot_indices.append((0, N - 2))
    _2q_plot_indices.append((0, N - 1))

    (
        s_output_path,
        s_evolution_path,
        s_estimation_path,
        s_simulation_path,
        s_plot_path,
    ) = experiment_routines.generate_paths()
    s_simulation_path += S_FILE_PREFIX
    load_files_prefix = ""
    if load_unique_id != "":
        load_files_prefix = s_simulation_path + "." + load_unique_id

    s_plot_path += S_FILE_PREFIX + f".N={N}"

    # -------------------------------------------------------------------------
    # Execution section
    topology = sim_metadata["topology"]
    b_device = False
    if "falcon" or "eagle" in topology:
        s_topology = topology
        b_device = True
    else:  # E.g., 'chain.E'
        s_topology = f"{N}.{topology}"
    device_map = topologies.coupling_maps.get(s_topology, None)
    qubit_coord = topologies.qubit_coordinates.get(s_topology, None)
    if device_map is not None and qubit_coord is not None:
        if b_device:
            qubits = json.loads(sim_metadata["qubits"])
            q_coord = [qubit_coord[qubit] for qubit in qubits]
            coupling_map = [
                (pair[0], pair[1])
                for pair in device_map
                if pair[0] in qubits and pair[1] in qubits
            ]
            c_map = [
                (qubits.index(pair[0]), qubits.index(pair[1])) for pair in coupling_map
            ]
        else:
            q_coord = qubit_coord
            c_map = device_map
        plot_device(N, c_map, q_coord, b_save_figures, s_plot_path)
    b_save_final_state = True
    if N > 10 and s_solver == "scipy":
        b_save_final_state = False

    solver_params = {
        "N": N,
        "b_unique_id": True,
        "metadata": str(sim_metadata).replace("\n", "\t"),
        "t_init": t_init,
        "t_final": t_final,
        "tau": tau,
        "g_0": g_0,
        "g_1": g_1,
        "g_2": g_2,
        "h_z": h_z,
        "J": J,
        "J_z": J_z,
        "init_product_state": init_product_state,
        "init_cz_gates": init_cz_gates,
        "apply_gates": apply_gates,
        "load_files_prefix": load_files_prefix,
        "1q_components": _1q_components,
        "1q_indices": _1q_indices,
        "2q_components": _2q_components,
        "2q_indices": _2q_indices,
        "output_files_prefix": s_simulation_path,
        "b_save_final_state": b_save_final_state,
        "output_step": 1,
        "force_rho_hermitian_step": force_rho_Hermitian_step,
    }
    if b_stabilizers:
        _3q_indexes = (
            experiment_routines.RING_3Q_XZZ_LIST
            if N == 12
            else experiment_routines.CHAIN_3Q_XZZ_LIST
        )
        _3q_indexes = _3q_indexes.copy()
        solver_params["3q_components"] = ["xzz"]
        solver_params["3q_indices"] = _3q_indexes

    _extend_parity_qubits(solver_params, parity_nu=parity_nu, parity_b=parity_b)
    promote_indices(_1q_plot_indices)
    promote_indices(_2q_plot_indices)
    if custom_observables is not None:
        for obs in custom_observables:
            promote_custom_obs_indices(obs[1])
        solver_params["custom_observables"] = custom_observables

    # Create the solver, and update solver-specific parameters
    if s_solver == "mpo":
        solver = LindbladMPOSolver()
        solver_params.update(
            {
                "max_dim_rho": sim_metadata["max_dim_rho"],
                "cut_off_rho": sim_metadata["cut_off_rho"],
            }
        )
    elif s_solver == "scipy":
        from lindbladmpo.examples.simulation_building.LindbladMatrixSolver import (
            LindbladMatrixSolver,
        )

        solver = LindbladMatrixSolver()
        solver_params.update(
            {
                "method": sim_metadata["method"],
                "atol": sim_metadata["atol"],
                "rtol": sim_metadata["rtol"],
            }
        )
    else:
        raise Exception("Solver type is unsupported.")

    # Verify parameters and create the solver input file. This will create also a unique id fot the
    # solver output files, that we store in the db.
    solver.build(solver_params)
    sim_metadata["unique_id"] = solver.parameters["unique_id"]

    if b_save_to_db:
        s_db_path = s_output_path + S_DB_FILENAME
        save_to_db(s_db_path, sim_metadata)

    # Solve, save final state and observable data files.
    solver.solve()

    # Plot figures
    s_file_prefix = s_plot_path + solver.s_id_suffix
    for s_obs_name in _1q_plot_components:
        s_title = f"$\\langle\\sigma^{s_obs_name}_j(t)\\rangle$"
        plot_1q_obs_curves(
            solver.parameters,
            solver.result,
            s_obs_name,
            _1q_plot_indices,
            fontsize=fontsize,
            b_save_figures=b_save_figures,
            s_file_prefix=s_file_prefix,
            s_title=s_title,
        )
    for s_obs_name in _2q_plot_components:
        s_title = f"$\\langle\\sigma^{s_obs_name[0]}_i\\sigma^{s_obs_name[1]}_j(t)\\rangle_{{c}}$"
        plot_2q_correlation_curves(
            solver.parameters,
            solver.result,
            s_obs_name,
            _2q_plot_indices,
            fontsize=fontsize,
            b_save_figures=b_save_figures,
            s_file_prefix=s_file_prefix,
            s_title=s_title,
        )


def plot_simulation_stabilizers(
    s_simulation_id,
    fontsize=16,
    b_save_figures=True,
    _3q_plot_indices=None,
):
    (
        s_output_path,
        s_evolution_path,
        s_estimation_path,
        s_simulation_path,
        s_plot_path,
    ) = experiment_routines.generate_paths()

    s_simulation_csv = s_output_path + S_DB_FILENAME
    parameters = get_simulation_dict(s_simulation_csv, s_simulation_id)
    s_output_file = s_simulation_path + S_FILE_PREFIX + "." + s_simulation_id
    result = LindbladMPOSolver.load_output(s_output_file)
    N = parameters["N"]

    s_file_prefix = (
        s_plot_path
        + experiment_routines.S_EVOLUTION_DATA_PREFIX
        + f".N={N}."
        + s_simulation_id
    )
    if not os.path.exists(s_plot_path):
        os.mkdir(s_plot_path)

    s_obs_name = "xzz"
    if _3q_plot_indices is None:
        _3q_plot_indices = (
            experiment_routines.RING_3Q_XZZ_LIST
            if N == 12
            else experiment_routines.CHAIN_3Q_XZZ_LIST
        )
    _3q_sim_plot_indices = _3q_plot_indices.copy()
    promote_indices(_3q_sim_plot_indices)
    ax, obs_data_list = plot_3q_obs_curves(
        parameters,
        result,
        s_obs_name,
        _3q_sim_plot_indices,
        fontsize=fontsize,
        b_save_figures=False,
        s_file_prefix=s_file_prefix,
    )
    times = None
    stab_norm = None
    for obs_data in obs_data_list:
        if times is None:
            times = obs_data[0]
            stab_norm = np.asarray(obs_data[1]) ** 2
        else:
            stab_norm += np.asarray(obs_data[1]) ** 2
    stab_norm = np.sqrt(stab_norm / len(obs_data_list))
    s_title = "Stabilizers norm"
    ax = plot_curves(
        [(times, stab_norm)], ["$\\|\\vec{S}\\|_2$"], s_title, fontsize=fontsize
    )
    _, _, t_tick_labels, _ = prepare_time_data(parameters)
    # ax.set_xticks(t_tick_indices)
    ax.set_xticks(t_tick_labels)
    ax.set_xticklabels(t_tick_labels, fontsize=fontsize)
    if b_save_figures:
        plt.savefig(s_file_prefix + "stabilizers.norm.png")


def plot_experiment_simulation(
    s_simulation_id,
    fontsize=16,
    b_mitigate_readout=True,
    b_correlations=False,
    b_stabilizers=True,
    b_save_figures=True,
    b_plot_z=True,
    _1q_plot_indices=None,
    _2q_plot_indices=None,
    _3q_plot_indices=None,
    b_mean_vector=False,
):
    (
        s_output_path,
        s_evolution_path,
        s_estimation_path,
        s_simulation_path,
        s_plot_path,
    ) = experiment_routines.generate_paths()

    s_simulation_csv = s_output_path + S_DB_FILENAME
    parameters = get_simulation_dict(s_simulation_csv, s_simulation_id)
    s_output_file = s_simulation_path + S_FILE_PREFIX + "." + s_simulation_id
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
    s_file_prefix = (
        s_plot_path
        + experiment_routines.S_EVOLUTION_DATA_PREFIX
        + f".N={N}."
        + s_simulation_id
    )
    if not os.path.exists(s_plot_path):
        os.mkdir(s_plot_path)

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
        physical_qubits,
        ev_1Q_dict,
        correlations_dict,
        stabilizers_dict,
    ) = load_observables(
        s_pickle_file,
        topo_index,
        xzz_list,
        b_mitigate_readout,
        b_correlations,
        p_0_given_0,
        p_0_given_1,
        b_stabilizers,
        b_plot_z,
    )
    times = times / time_unit + 1 * t_gates

    if b_stabilizers:
        s_obs_name = "xzz"
        if _3q_plot_indices is None:
            _3q_plot_indices = (
                experiment_routines.RING_3Q_XZZ_LIST
                if N == 12
                else experiment_routines.CHAIN_3Q_XZZ_LIST
            )
        _3q_sim_plot_indices = _3q_plot_indices.copy()
        promote_indices(_3q_sim_plot_indices)
        if _1q_plot_indices is None:
            _1q_plot_indices = (
                experiment_routines.RING_TOPOLOGY_INDEXES
                if N == 12
                else experiment_routines.CHAIN_3Q_TOPOLOGY_INDEXES
            )
        _1q_sim_plot_indices = _3q_plot_indices.copy()
        promote_indices(_1q_sim_plot_indices)
        stab = stabilizers_dict
        for i_x_qubit, _3q_tuple in zip(_1q_plot_indices, _3q_plot_indices):
            i, j, k = _3q_tuple[0], _3q_tuple[1], _3q_tuple[2]
            q1, q2, q3 = physical_qubits[i], physical_qubits[j], physical_qubits[k]
            x1 = physical_qubits[i_x_qubit]
            fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 7))
            stab_v = stab[f"stabilizer_{i}"]
            XZZ = unp_n(stab_v)
            XZZ_err = unp_s(stab_v)
            ax.errorbar(
                times, XZZ, XZZ_err, fmt="bo", alpha=0.5, capsize=4, markersize=5
            )
            s_title = ""
            plot_3q_obs_curves(
                parameters,
                result,
                s_obs_name,
                [(sim_qubits[i], sim_qubits[j], sim_qubits[k])],
                ax,
                fontsize=fontsize,
                b_save_figures=False,
                s_title=s_title,
            )
            s_stab = f"$\\langle X_{{{q1}}} Z_{{{q2}}} Z_{{{q3}}}\\rangle$"
            plot_1q_obs_curves(
                parameters,
                result,
                "x",
                [sim_qubits[i_x_qubit]],
                ax,
                line_styles=["--"],
                fontsize=fontsize,
                b_save_figures=False,
                s_title=s_title,
            )
            s_x1 = f"$\\langle X_{{{x1}}}\\rangle$"
            # plot_1q_obs_curves(
            #     parameters,
            #     result,
            #     "y",
            #     [sim_qubits[i_x_qubit]],
            #     ax,
            #     line_styles=[":"],
            #     fontsize=fontsize,
            #     b_save_figures=False,
            #     s_title=s_title,
            # )
            # s_y1 = f"$\\langle Y_{{{x1}}}\\rangle$"
            ax.legend(
                [s_stab + ", sim", s_x1 + ", sim", s_stab + ", exp"],
                loc="upper right",
                frameon=False,
            )
            if abs(time_unit - 1e-6) < 1e-8:
                ax.set_xlabel(r"$Delay [\mu s]$")
            if b_save_figures:
                plt.savefig(s_file_prefix + ".xyz" + f".Q{q1}Q{q2}Q{q3}.png")

        plot_simulation_stabilizers(s_simulation_id, fontsize)
        return

    for i_plot_qubit, qubit in enumerate(physical_qubits):
        RamX_data = unp_n(ev_1Q_dict[f"X_{qubit}"])
        RamX_err = unp_s(ev_1Q_dict[f"X_{qubit}"])
        RamY_data = unp_n(ev_1Q_dict[f"Y_{qubit}"])
        RamY_err = unp_s(ev_1Q_dict[f"Y_{qubit}"])
        if b_plot_z:
            RamZ_data = unp_n(ev_1Q_dict[f"Z_{qubit}"])
            RamZ_err = unp_s(ev_1Q_dict[f"Z_{qubit}"])
        if i_plot_qubit == 0:
            x_mean_exp = np.copy(RamX_data)
            y_mean_exp = np.copy(RamY_data)
            x_var_exp = RamX_err**2
            y_var_exp = RamY_err**2
        else:
            x_mean_exp += np.copy(RamX_data)
            y_mean_exp += np.copy(RamY_data)
            x_var_exp += RamX_err**2
            y_var_exp += RamY_err**2
    x_mean_exp = x_mean_exp / N
    y_mean_exp = y_mean_exp / N
    x_err_exp = (x_var_exp**0.5) / N
    y_err_exp = (y_var_exp**0.5) / N

    if _1q_plot_indices is None:
        _1q_plot_indices = np.asarray(range(N))
    n_plot_qubits = len(_1q_plot_indices)
    axs = None
    for i_plot_qubit in range(n_plot_qubits):
        qubit = physical_qubits[_1q_plot_indices[i_plot_qubit]]
        RamX_data = unp_n(ev_1Q_dict[f"X_{qubit}"])
        RamX_err = unp_s(ev_1Q_dict[f"X_{qubit}"])
        RamY_data = unp_n(ev_1Q_dict[f"Y_{qubit}"])
        RamY_err = unp_s(ev_1Q_dict[f"Y_{qubit}"])
        b_plot_rho = True
        if b_plot_rho:
            rho_data = unp.sqrt(
                ev_1Q_dict[f"X_{qubit}"] ** 2 + ev_1Q_dict[f"Y_{qubit}"] ** 2
            )
            fig_rho, ax_rho = plt.subplots(1, 1)
            ax_rho.errorbar(
                times,
                unp_n(rho_data),
                yerr=unp_s(rho_data),
                fmt="ro",
                alpha=0.85,
                capsize=4,
                markersize=5,
            )

        if b_plot_z:
            import qutip

            RamZ_data = unp_n(ev_1Q_dict[f"Z_{qubit}"])
            RamZ_err = unp_s(ev_1Q_dict[f"Z_{qubit}"])

            fig_bloch = plt.figure()
            ax_bloch = Axes3D(fig_bloch, azim=-40, elev=30)
            sphere = qutip.Bloch(fig=fig_bloch)
            length = len(RamX_data)
            nrm = mpl.colors.Normalize(0, length)
            colors = cm.hot(nrm(range(length)))

            sphere.point_color = list(colors)

            sphere.add_points([RamX_data, RamY_data, RamZ_data], meth="m")
            sphere.add_points([RamX_data, RamY_data, RamZ_data], meth="l")
            sphere.make_sphere()

            """
            # to make animation:
            def animate(i):
                sphere.clear()
                # sphere.add_vectors([np.sin(theta), 0, np.cos(theta)])
                sphere.add_points([sx[:i + 1], sy[:i + 1], sz[:i + 1]], meth='m')
                sphere.add_points([sx[:i + 1], sy[:i + 1], sz[:i + 1]], meth='l')
                sphere.make_sphere()
                return ax
            ani = animation.FuncAnimation(fig, animate, frames=length,
                                             blit=False, repeat=False, interval=1)
            ani.save('animation.gif', fps=20)
            """

        i_axis = 0
        i_x_label_axis = 0
        n_fig_qubits = 1
        b_single_q_per_plot = True
        if (
            i_plot_qubit < (n_plot_qubits - 1) or (n_plot_qubits % 2) == 0
        ) and not b_single_q_per_plot:
            i_x_label_axis = 1
            n_fig_qubits += 1
        if (i_plot_qubit % 2) == 0 or b_single_q_per_plot:
            s_filename = f".x.y.Q{qubit}"
            s_title = ""  # "Simulation and Experiment, 1Q"
            fig, axs = plt.subplots(n_fig_qubits, 1, sharex=True, figsize=(11, 7))
            if n_fig_qubits == 1:
                axs = [axs]
        else:
            s_filename += f".Q{qubit}"
            s_title = ""
            i_axis += 1
        ax = axs[i_axis]
        ax.errorbar(
            times,
            RamX_data,
            yerr=RamX_err,
            fmt="bo",
            alpha=0.85,
            capsize=4,
            markersize=5,
        )
        plot_1q_obs_curves(
            parameters,
            result,
            "x",
            [sim_qubits[_1q_plot_indices[i_plot_qubit]]],
            ax=ax,
            fontsize=fontsize,
            b_save_figures=False,
            s_title=s_title,
            b_legend_labels=False,
        )
        ax.errorbar(
            times,
            RamY_data,
            yerr=RamY_err,
            fmt="r+",
            alpha=0.85,
            capsize=4,
            markersize=5,
        )
        plot_1q_obs_curves(
            parameters,
            result,
            "y",
            [sim_qubits[_1q_plot_indices[i_plot_qubit]]],
            ax=ax,
            fontsize=fontsize,
            b_save_figures=False,
            s_file_prefix=s_file_prefix,
            s_title=s_title,
            b_legend_labels=False,
        )
        s_x = f"$\\langle X_{{{qubit}}}\\rangle$"
        s_y = f"$\\langle Y_{{{qubit}}}\\rangle$"
        if b_plot_z:
            ax.errorbar(
                times,
                RamZ_data,
                yerr=RamZ_err,
                fmt="gx",
                alpha=0.85,
                capsize=4,
                markersize=5,
            )
            plot_1q_obs_curves(
                parameters,
                result,
                "z",
                [sim_qubits[_1q_plot_indices[i_plot_qubit]]],
                ax=ax,
                fontsize=fontsize,
                b_save_figures=False,
                s_file_prefix=s_file_prefix,
                s_title=s_title,
                b_legend_labels=False,
            )
            s_z = f"$\\langle Z_{{{qubit}}}\\rangle$"
            legends = [
                s_x + ", sim",
                s_y + ", sim",
                s_z + ", sim",
                s_x + ", exp",
                s_y + ", exp",
                s_z + ", exp",
            ]
        else:
            legends = [s_x + ", sim", s_y + ", sim", s_x + ", exp", s_y + ", exp"]

        ax.legend(legends, loc="upper right", frameon=False)
        if i_axis == i_x_label_axis:
            if abs(time_unit - 1e-6) < 1e-8:
                ax.set_xlabel(r"$Delay [\mu s]$")
            if b_save_figures:
                plt.savefig(s_file_prefix + s_filename + ".png")

    x_values = np.array(
        [
            [
                result["obs-1q"][("x", (i,))][1][t]
                for t in range(len(result["obs-1q"][("x", (i,))][0]))
            ]
            for i in sim_qubits
        ]
    )
    y_values = np.array(
        [
            [
                result["obs-1q"][("y", (i,))][1][t]
                for t in range(len(result["obs-1q"][("y", (i,))][0]))
            ]
            for i in sim_qubits
        ]
    )
    # z_values = np.array([[result['obs-1q'][('z', (i,))][1][t]
    # 					  for t in range(len(result['obs-1q'][('z', (i,))][0]))] for i in sim_qubits])
    n_steps = x_values.shape[1]
    t_eval = np.linspace(t_init, t_final, n_steps)

    n_steps = x_values.shape[1]
    x_mean = [np.mean(x_values[:, i]) for i in range(n_steps)]
    y_mean = [np.mean(y_values[:, i]) for i in range(n_steps)]

    if b_mean_vector:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(11, 7))
        s_title = "Mean Bloch vector"
        s_x = "$ N^{-1}\sum_i \\langle X_i \\rangle$"
        s_y = "$ N^{-1}\sum_i \\langle Y_i \\rangle$"
        ax.errorbar(
            times,
            x_mean_exp,
            yerr=x_err_exp,
            fmt="bo",
            alpha=0.85,
            capsize=4,
            markersize=5,
            label=s_x + ", exp",
        )
        ax.errorbar(
            times,
            y_mean_exp,
            yerr=y_err_exp,
            fmt="ro",
            alpha=0.85,
            capsize=4,
            markersize=5,
            label=s_y + ", exp",
        )
        ax.legend()  # loc = 'upper right', frameon = False)
        if abs(time_unit - 1e-6) < 1e-8:
            ax.set_xlabel(r"$Delay [\mu s]$")
        ax.set_title(s_title)
        plt.rcParams.update({"font.size": fontsize})
        plt.plot(t_eval, x_mean, label=s_x + ", sim")
        plt.plot(t_eval, y_mean, label=s_y + ", sim")
        plt.legend(fontsize=fontsize)
        if b_save_figures:
            plt.savefig(s_file_prefix + ".x.y.mean" + ".png")

    if _2q_plot_indices is None:
        _2q_plot_indices = [(0, 1)]
        if N > 2:
            _2q_plot_indices.extend([(1, 2)])
        if N > 3:
            _2q_plot_indices.extend([(0, 2)])
            _2q_plot_indices.extend([(2, 3)])
            _2q_plot_indices.extend([(1, 3)])
        if N > 4:
            _2q_plot_indices.extend([(2, 4)])
        if N > 5:
            _2q_plot_indices.extend([(2, 5)])

    if b_correlations:
        corr = correlations_dict
        for pair in _2q_plot_indices:
            i = pair[0]
            j = pair[1]
            q1 = physical_qubits[i]
            q2 = physical_qubits[j]
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
            corr_v = corr[f"X_{q1}_X_{q2}_c"]
            XX = unp_n(corr_v)
            XX_err = unp_s(corr_v)
            ax = axs[0]
            ax.errorbar(times, XX, XX_err, fmt="bo", alpha=0.5, capsize=4, markersize=5)
            s_obs_name = "xx"
            s_title = ""
            plot_2q_correlation_curves(
                parameters,
                result,
                s_obs_name,
                [(sim_qubits[i], sim_qubits[j])],
                ax,
                fontsize=fontsize,
                b_save_figures=False,
                s_file_prefix=s_file_prefix,
                s_title=s_title,
            )
            s_corr = f"$\\langle X_{{{q1}}} X_{{{q2}}}\\rangle_c$"
            ax.legend(
                [s_corr + ", sim", s_corr + ", exp"], loc="upper right", frameon=False
            )

            corr_v = corr[f"Y_{q1}_Y_{q2}_c"]
            YY = unp_n(corr_v)
            YY_err = unp_s(corr_v)
            ax = axs[1]
            ax.errorbar(
                times, YY, yerr=YY_err, fmt="bo", alpha=0.5, capsize=4, markersize=5
            )
            s_obs_name = "yy"
            plot_2q_correlation_curves(
                parameters,
                result,
                s_obs_name,
                [(sim_qubits[i], sim_qubits[j])],
                ax,
                fontsize=fontsize,
                b_save_figures=False,
                s_file_prefix=s_file_prefix,
                s_title=s_title,
            )
            s_corr = f"$\\langle Y_{{{q1}}} Y_{{{q2}}}\\rangle_c$"
            ax.legend(
                [s_corr + ", sim", s_corr + ", exp"], loc="upper right", frameon=False
            )

            if abs(time_unit - 1e-6) < 1e-8:
                ax.set_xlabel(r"$Delay [\mu s]$")
            if b_save_figures:
                plt.savefig(s_file_prefix + ".xx.yy" + f".Q{q1}Q{q2}.png")
