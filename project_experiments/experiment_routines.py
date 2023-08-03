# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Routines for managing the research project running multiple estimation and evolution experiments.
"""

import os.path
from datetime import datetime
from lindbladmpo.examples.qubit_driving.output_routines import *
from lindbladmpo.LindbladMPOSolver import *
from project_experiments.library.bayesian.spam_1q_builder import BayesianSPAMBuilder
from project_experiments.library.bayesian.spam_1q_estimator import BayesianSPAMEstimator

from project_experiments.library.multiqubit_xy import MultiqubitXY
from project_experiments.library.parity_oscillations import ParityOscillations
from project_experiments.library.zz_ramsey import ZZRamsey
from project_experiments.load_routines import unpack_analysis_result
from qiskit_experiments.library import T1
from qiskit_experiments.framework.composite import BatchExperiment, ParallelExperiment
from project_experiments.partition import partition_qubit_pairs
from project_experiments.partition import partition_qubits

b_save_experiments = False
"""Set to true in order to save qiskit-experiments to a database service"""

S_EVOLUTION_DF_FILENAME = "evolution.experiments.csv"
"""File name to use for the database of all evolution experiments."""

S_EVOLUTION_DATA_PREFIX = "evolution"
"""Prefix for the file names of all evolution output files."""

S_ESTIMATION_DATA_PREFIX = "estimation"
"""File name prefix to use for the database of an estimation experiments."""

N_SPAM_SHOTS = 8000
"""Number of shots requested for SPAM estimation circuits."""

RING_TOPOLOGY_INDEXES = [0, 1, 3, 5, 7, 9, 11, 10, 8, 6, 4, 2]
"""Indexes of qubits in a 12Q ring when ordered in an MPO zigzag configuration."""

RING_ZZ_DD_GROUP_INDEXES = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
"""Group index for ZZ DD CPMG sequences (odd/even qubits along the ring)."""

RING_3Q_XZZ_LIST = [
    (
        RING_TOPOLOGY_INDEXES[i],
        RING_TOPOLOGY_INDEXES[np.mod(i - 1, 12)],
        RING_TOPOLOGY_INDEXES[np.mod(i + 1, 12)],
    )
    for i in range(12)
]
"""Qubits indexes of nearest neighbors for <ZXZ> stabilizer 3Q-observables in a 12Q ring."""

CHAIN_3Q_ZZ_DD_GROUP_INDEXES = [0, 1, 0]
"""Group index for ZZ DD CPMG sequences (odd/even qubits along the chain)."""

CHAIN_3Q_TOPOLOGY_INDEXES = [0, 1, 2]
"""Indexes of qubits in a 3Q chain."""

CHAIN_3Q_XZZ_LIST = [(1, 0, 2)]
"""Qubits indexes of nearest neighbors for <ZXZ> stabilizer 3Q-observables in a 3Q chain."""


def print_log(s_message: str):
    """Logs messages to the log file (and console), with a time stamp."""
    now = datetime.now().astimezone()
    s_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    s_message = f"{s_date_time} - " + s_message
    print(s_message)


def generate_paths():
    """Concatenate a data directory and figures directory path, and create the directories.

    Returns:
            A 5-tuple with the directory strings.
    """
    s_output_path = (
        os.path.abspath("../output") + "/"
    )  # use the absolute path of the current file
    if not os.path.exists(s_output_path):
        os.mkdir(s_output_path)
    s_evolution_path = s_output_path + "evolutions/"
    if not os.path.exists(s_evolution_path):
        os.mkdir(s_evolution_path)
    s_estimation_path = s_output_path + "estimations/"
    if not os.path.exists(s_estimation_path):
        os.mkdir(s_estimation_path)
    s_simulation_path = s_output_path + "simulations/"
    if not os.path.exists(s_simulation_path):
        os.mkdir(s_simulation_path)
    s_plot_path = s_output_path + "figures/"
    if not os.path.exists(s_plot_path):
        os.mkdir(s_plot_path)
    return (
        s_output_path,
        s_evolution_path,
        s_estimation_path,
        s_simulation_path,
        s_plot_path,
    )


def run_experiment(
    edge_groups_name: str,
    parity_offset_freq: float,
    evolution_offset_freq: float,
    t_final: float,
    backend,
    init_state: str,
    delays_T1,
    delays_parity,
    parity_delays_cut_off,
    delays_zz_ramsey,
    osc_freq_zz_ramsey,
    delays_evolution,
    shots=1024,
    b_stabilizers=False,
    n_dd_cycles=0,
    b_just_evolution=False,
    b_zz_dd=False,
    b_dilute_evolution=False,
    b_add_z_measurement=False,
    b_estimation_after_evolution=True,
    b_nu_estimation_in_single_job=True,
    load_jobs=[],
):
    (
        all_qubits,
        estimation_qubits,
        estimation_pairs,
        evolution_groups,
    ) = get_experiment_edges(edge_groups_name, backend)

    s_backend = backend.name if isinstance(backend.name, str) else backend.name()
    s_topology = (
        f"{backend.configuration().n_qubits}."
        f"{backend.configuration().processor_type['family'].lower()}"
    )
    print_log(f"Preparing experiment on backend: {s_backend}.")
    s_output_path, s_evolution_path, s_estimation_path, _, _ = generate_paths()

    T1_exp = []
    for group in all_qubits:
        exps = []
        for qubits in group:
            exp = T1(physical_qubits=qubits, delays=delays_T1)
            exps.append(exp)
        par_exp = ParallelExperiment(exps)
        T1_exp.append(par_exp)

    gates = BayesianSPAMEstimator.BAYESIAN_QPCM_GATES
    parameters = BayesianSPAMEstimator.BAYESIAN_QPCM_PARAMETERS
    prior_intervals = BayesianSPAMEstimator.BAYESIAN_QPCM_PRIORS
    n_repeats = int(np.ceil(N_SPAM_SHOTS / shots))
    spam_model = BayesianSPAMBuilder(
        gates,
        parameters,
        prior_intervals,
        n_draws=int(10e6),
        n_repeats=n_repeats,
        user_qubit_groups=all_qubits,
    )
    SPAM_exp = spam_model.build(backend)

    qubit_estimation_exp = []
    if delays_zz_ramsey is None or len(delays_zz_ramsey) == 0:
        raise Exception("delays_zz_ramsey must be nonempty.")

    estimation_funcs = [
        lambda qubit: ParityOscillations(
            qubit=qubit,
            delays=delays_parity,
            osc_freq=parity_offset_freq,
            cut_off_delay=parity_delays_cut_off,
            fixed_phase=False,
        ),
    ]
    for group in estimation_qubits:
        print_log("Preparing estimation experiments for group: " + str(group))
        for i, exp_func in enumerate(estimation_funcs):
            exps = []
            for qubit in group:
                exp = exp_func(qubit[0])
                exps.append(exp)
            par_exp = ParallelExperiment(exps)
            par_exp.set_transpile_options(
                scheduling_method="alap", optimization_level=0
            )
            qubit_estimation_exp.append(par_exp)

    estimation_edge_funcs = [
        lambda qubits: ZZRamsey(
            qubit=qubits, delays=delays_zz_ramsey, osc_freq=osc_freq_zz_ramsey
        ),
    ]

    edge_estimation_exp = []
    for group in estimation_pairs:
        print_log("Preparing estimation experiments for group: " + str(group))
        for exp_func in estimation_edge_funcs:
            exps = []
            for qubit_pair in group:
                exp = exp_func(qubit_pair)
                exps.append(exp)
            par_exp = ParallelExperiment(exps)
            par_exp.set_transpile_options(
                scheduling_method="alap", optimization_level=0
            )
            edge_estimation_exp.append(par_exp)

    evolution_exp = []
    evolution_metadatas = []
    evolution_experiments = []
    if isinstance(n_dd_cycles, int):
        n_dd_cycles = [n_dd_cycles]
    for _n_dd_cycles in n_dd_cycles:
        dilute_evolutions = [False]
        if b_dilute_evolution and _n_dd_cycles:
            dilute_evolutions.append(True)
        for b_dilute in dilute_evolutions:
            delays = delays_evolution.copy()
            if b_dilute:
                delays = delays[::4]
            i_qubits_group = 0
            for i_group, group in enumerate(evolution_groups):
                print_log(f"Preparing evolution experiment for group #{i_group}")
                exps = []
                for qubits in group:
                    i_qubits_group += 1
                    N = len(qubits)
                    topology_index = (
                        RING_TOPOLOGY_INDEXES if N == 12 else CHAIN_3Q_TOPOLOGY_INDEXES
                    )
                    zz_dd_group_indexes = (
                        RING_ZZ_DD_GROUP_INDEXES
                        if N == 12
                        else CHAIN_3Q_ZZ_DD_GROUP_INDEXES
                    )
                    s_uuid = uuid.uuid4().hex
                    s_file_path = (
                        s_evolution_path + S_EVOLUTION_DATA_PREFIX + "." + s_uuid
                    )
                    pair_groups = partition_qubit_pairs(
                        backend, 2, multigraph=False, node_subset=qubits
                    )
                    exp = MultiqubitXY(
                        qubits=qubits,
                        delays=delays,
                        osc_freq=evolution_offset_freq,
                        file_path_prefix=s_file_path,
                        s_init_state=init_state,
                        pair_groups=pair_groups,
                        storage=dict(),
                        b_stabilizers=b_stabilizers,
                        topology_index=topology_index,
                        n_dd_cycles=_n_dd_cycles,
                        b_zz_dd=b_zz_dd,
                        zz_dd_group_indexes=zz_dd_group_indexes,
                        b_add_z_measurement=b_add_z_measurement,
                    )
                    exps.append(exp)
                    exp_metadata = {
                        "backend": s_backend,
                        "n_qubits": N,
                        "i_qubits_group": i_qubits_group,
                        "unique_id": s_uuid,
                        "max_chi_sq": "",
                        "median_chi_sq": "",
                        "init_state": init_state,
                        "logical_gate": "",
                        "n_dd_cycles": _n_dd_cycles,
                        "b_zz_dd": b_zz_dd,
                        "edges_descr": edge_groups_name,
                        "topology": s_topology,
                        "b_stabilizers": b_stabilizers,
                        "t_final": t_final,
                        "offset_freq": evolution_offset_freq,
                        "qubits": qubits,
                        "cz_groups": pair_groups,
                        "gate_len_groups": "",
                        "zz_dd_group_indexes": zz_dd_group_indexes,
                        "delays": delays,
                        "estimation_id": "",
                        "experiment_id": "",
                        "experiment_link": "",
                    }
                    evolution_metadatas.append(exp_metadata)
                    evolution_experiments.append(exp)
                par_exp = ParallelExperiment(exps)
                par_exp.set_transpile_options(
                    scheduling_method="alap", optimization_level=0
                )
                evolution_exp.append(par_exp)

    exp_list = [BatchExperiment(edge_estimation_exp)]
    if b_nu_estimation_in_single_job:
        exp_list.append(BatchExperiment(qubit_estimation_exp))
    else:
        for exp in qubit_estimation_exp:
            exp_list.append(exp)
    exp_list.append(BatchExperiment(T1_exp + SPAM_exp + evolution_exp))

    if b_estimation_after_evolution:
        qubit_estimation_after_evolution_exp = []
        estimation_funcs_after_evolution = [
            lambda qubit: ParityOscillations(
                qubit=qubit,
                delays=delays_parity,
                osc_freq=parity_offset_freq,
                cut_off_delay=parity_delays_cut_off,
                fixed_phase=False,
                parameters_suffix="_after_evolution",
            ),
        ]
        for group in estimation_qubits:
            print_log(
                "Preparing estimation after evolution experiments for group: "
                + str(group)
            )
            for i, exp_func in enumerate(estimation_funcs_after_evolution):
                exps = []
                for qubit in group:
                    exp = exp_func(qubit[0])
                    exps.append(exp)
                par_exp = ParallelExperiment(exps)
                par_exp.set_transpile_options(
                    scheduling_method="alap", optimization_level=0
                )
                qubit_estimation_after_evolution_exp.append(par_exp)
        if b_nu_estimation_in_single_job:
            exp_list.append(BatchExperiment(qubit_estimation_after_evolution_exp))
        else:
            for exp in qubit_estimation_after_evolution_exp:
                exp_list.append(exp)

    if b_just_evolution:
        exp_list = evolution_exp
    final_exp = BatchExperiment(exp_list, flatten_results=True)
    b_multijob = True  # currently the only option
    if b_multijob:
        final_exp.set_experiment_options(separate_jobs=True)

    print_log("Executing experiment.")
    final_exp.set_transpile_options(
        timing_constraints=backend.configuration().timing_constraints,
        scheduling_method="alap",
        optimization_level=0,
    )
    data = []
    experiment_id = ""
    if load_jobs:
        final_exp._set_backend(backend)
        final_exp._finalize()
        # Generate and transpile circuits
        transpiled_circuits = final_exp._transpiled_circuits()
        # Initialize result container
        expdata = final_exp._initialize_experiment_data()
        print("Loading jobs")
        # service = ExperimentData.get_service_from_backend(backend)
        retrieved_jobs = []
        for job_id in load_jobs:
            retrieved_job = backend.provider.backend.retrieve_job(job_id)
            retrieved_jobs.append(retrieved_job)
        # expdata = ExperimentData(service=service, experiment=final_exp, backend=backend)
        expdata.add_jobs(retrieved_jobs)
        print("Finished loading jobs")
        exp_data = final_exp.analysis.run(expdata)
    else:
        exp_data = final_exp.run(backend=backend, shots=shots)
    exp_data.block_for_results()

    # ---- set figure titles -----
    for figkey in range(len(exp_data.figure_names)):
        fig = exp_data.figure(figkey)
        old_title = fig.figure.axes[0].get_title()
        new_title = old_title
        for i, qubit in enumerate(fig.metadata["qubits"]):
            new_title = f"q{i + 1}=Q{qubit}  " + new_title
        fig.figure.axes[0].set_title(new_title)

    if b_save_experiments:
        exp_data.share_level = "project"
        exp_data.tags = ["graph_state_dynamics"]
        exp_data.save()

    # -- save fit parameters --
    analysis_results = exp_data.analysis_results()
    t = datetime.now()
    for result in analysis_results:
        data_dict, variables_name = unpack_analysis_result(
            result, backend.name if isinstance(backend.name, str) else backend.name(), t
        )
        if data_dict is not None:
            data.append(data_dict)
            experiment_id = data_dict["experiment_id"]
    print(f"Experiment completed, analyzed and saved.")

    s_link = ""
    if b_save_experiments:
        # Use experiment_id here to generate a link to the saved experiment
        # s_link = f"https://quantum-computing.ibm.com/experiments/{experiment_id}?plotsLimit=50"
        pass

    df = pd.DataFrame(data)
    s_uuid = uuid.uuid4().hex
    s_file_path = s_estimation_path + S_ESTIMATION_DATA_PREFIX + "." + s_uuid
    df.to_csv(s_file_path + ".csv", index=False)

    for exp, exp_metadata in zip(evolution_experiments, evolution_metadatas):
        qubits = exp_metadata["qubits"]
        delays = exp_metadata.get("delays", "")
        pair_groups = exp_metadata.get("cz_groups", "")
        gate_len_groups = exp.storage.get("gate_len_groups")
        zz_dd_group_indexes = exp_metadata.get("zz_dd_group_indexes", "")
        s_qubits = ""
        for qubit in qubits:
            s_qubits += f"'Q{qubit}',"
        s_qubits = s_qubits[:-1]
        chi_sq = df.query(
            f"device_components in [{s_qubits}]" f"and variable_name != 'T1'"
        )["chi_sq"]
        max_chi_sq = chi_sq.max()
        median_chi_sq = chi_sq.median()

        exp_metadata.update(
            {
                "qubits": str(qubits),
                "delays": str(list(delays)),
                "cz_groups": str(pair_groups),
                "gate_len_groups": str(gate_len_groups),
                "zz_dd_group_indexes": str(zz_dd_group_indexes),
                "median_chi_sq": round(median_chi_sq, 2),
                "max_chi_sq": round(max_chi_sq, 2),
                "estimation_id": s_uuid,
                "experiment_id": experiment_id,
                "experiment_link": s_link,
            }
        )
        s_db_path = s_output_path + S_EVOLUTION_DF_FILENAME
        save_to_db(s_db_path, exp_metadata)
    print_log("Experiments saved to local files. Terminating.")


def sort_groups_by_nu(groups, nu_freqs):

    # sort pairs by nu of target
    for group in groups:
        group.sort(
            key=lambda q_pair: nu_freqs[q_pair[1]], reverse=True
        )  # Highest nu first

    return groups


def sort_groups_by_control_target(unsorted_groups, backend):
    physical_groups = []
    properties = backend.properties()

    # set each pair in [control, target] order
    for group in unsorted_groups:
        physical_group = []
        for qubit_pair in group:
            i = qubit_pair[0]
            j = qubit_pair[1]
            inverted_pair = [qubit_pair[1], qubit_pair[0]]
            original_gate_len = properties.gate_length("cx", qubit_pair)
            inverted_gate_len = properties.gate_length("cx", inverted_pair)
            if inverted_gate_len > original_gate_len:
                physical_pair = [i, j]
            else:
                physical_pair = [j, i]
            physical_group.append(physical_pair)
        physical_groups.append(physical_group)

    return physical_groups


def get_experiment_edges(name: str, backend):
    node_subset = None
    estimation_pairs = None
    n_qubits = backend.configuration().num_qubits
    if name == "pairs":
        if n_qubits == 7:
            estimation_pairs = [[[0, 1], [4, 5]], [[1, 2], [5, 6]]]
            evolution_groups = [[[0, 1], [4, 5]], [[1, 2], [5, 6]]]
        else:
            estimation_pairs = [
                [
                    [0, 1],
                    [6, 7],
                    [12, 15],
                    [21, 23],
                    [3, 5],
                    [11, 14],
                    [19, 20],
                    [25, 26],
                ]
            ]
            evolution_groups = estimation_pairs.copy()
    elif name == "triples":
        if n_qubits == 7:
            estimation_pairs = [[[0, 1], [4, 5]], [[1, 2], [5, 6]]]
            evolution_groups = [[[0, 1, 2]], [[4, 5, 6]]]
        elif n_qubits == 127:
            evolution_groups = [
                [
                    [0, 1, 2],
                    [4, 5, 6],
                    [8, 9, 10],
                    [22, 23, 24],
                    [30, 31, 32],
                    [37, 38, 39],
                    [41, 42, 43],
                    [45, 46, 47],
                    [49, 50, 51],
                    [56, 57, 58],
                    [60, 61, 62],
                    [64, 65, 66],
                    [68, 69, 70],
                    [77, 78, 79],
                    [85, 86, 87],
                ]
            ]
            node_subset = evolution_groups[0][0].copy()
            for i_node in range(len(evolution_groups[0])):
                node_subset.extend(evolution_groups[0][i_node])
        else:  # 27 qubits
            estimation_pairs = [
                [[0, 1], [10, 12], [21, 23], [16, 19], [5, 8]],
                [[1, 2], [12, 13], [23, 24], [19, 22], [8, 9]],
            ]
            evolution_groups = [
                [[0, 1, 2], [10, 12, 13], [21, 23, 24], [16, 19, 22], [5, 8, 9]]
            ]
    elif name == "chain.7.A":
        evolution_groups = [[[3, 5, 8, 11, 14, 16, 19], [10, 12, 15, 18, 21, 23, 24]]]
        node_subset = evolution_groups[0][0].copy()
        node_subset.extend(evolution_groups[0][1])
    elif name == "chain.7.B":
        evolution_groups = [[[3, 5, 8, 11, 14, 16, 19], [1, 4, 7, 10, 12, 15, 18]]]
        node_subset = evolution_groups[0][0].copy()
        node_subset.extend(evolution_groups[0][1])
    elif name == "chain.7.C":
        evolution_groups = [
            [
                [0, 1, 2, 3, 5, 8, 9],
                [6, 7, 10, 12, 15, 18, 17],
                [23, 24, 25, 22, 19, 16, 14],
            ]
        ]
        node_subset = evolution_groups[0][0].copy()
        node_subset.extend(evolution_groups[0][1])
        node_subset.extend(evolution_groups[0][2])
    elif name == "rings.27.A":
        node_subset = [1, 4, 2, 7, 3, 10, 5, 12, 8, 13, 11, 14]
        evolution_groups = [[node_subset]]
    elif name == "rings.27.B":
        node_subset = [12, 15, 13, 18, 14, 21, 16, 23, 19, 24, 22, 25]
        evolution_groups = [[node_subset]]
    elif name == "rings.27":
        evolution_groups = [
            [[1, 4, 2, 7, 3, 10, 5, 12, 8, 13, 11, 14]],
            [[12, 15, 13, 18, 14, 21, 16, 23, 19, 24, 22, 25]],
        ]
        node_subset = evolution_groups[0][0].copy()
        node_subset.extend(evolution_groups[1][0])
    elif name == "rings.127.A":
        evolution_groups = [
            [
                [0, 1, 14, 2, 18, 3, 19, 4, 20, 15, 21, 22],
                [24, 25, 34, 26, 43, 27, 44, 28, 45, 35, 46, 47],
                [37, 38, 52, 39, 56, 40, 57, 41, 58, 53, 59, 60],
                [62, 63, 72, 64, 81, 65, 82, 66, 83, 73, 84, 85],
                [75, 76, 90, 77, 94, 78, 95, 79, 96, 91, 97, 98],
                [100, 101, 110, 102, 118, 103, 119, 104, 120, 111, 121, 122],
            ]
        ]
        node_subset = []
        for group in evolution_groups[0]:
            node_subset.extend(group)
    elif name == "rings.127.B":
        evolution_groups = [
            [
                [4, 5, 15, 6, 22, 7, 23, 8, 24, 16, 25, 26],
                [28, 29, 35, 30, 47, 31, 48, 32, 49, 36, 50, 51],
                [41, 42, 53, 43, 60, 44, 61, 45, 62, 54, 63, 64],
                [66, 67, 73, 68, 85, 69, 86, 70, 87, 74, 88, 89],
                [79, 80, 91, 81, 98, 82, 99, 83, 100, 92, 101, 102],
                [104, 105, 111, 106, 122, 107, 123, 108, 124, 112, 125, 126],
            ]
        ]
        node_subset = []
        for group in evolution_groups[0]:
            node_subset.extend(group)
    elif name == "rings.127.C":
        evolution_groups = [
            [
                [0, 1, 14, 2, 18, 3, 19, 4, 20, 15, 21, 22],
                [24, 25, 34, 26, 43, 27, 44, 28, 45, 35, 46, 47],
                [37, 38, 52, 39, 56, 40, 57, 41, 58, 53, 59, 60],
            ]
        ]
        node_subset = []
        for group in evolution_groups[0]:
            node_subset.extend(group)
    elif name == "rings.127.D":
        evolution_groups = [
            [
                [62, 63, 72, 64, 81, 65, 82, 66, 83, 73, 84, 85],
                [75, 76, 90, 77, 94, 78, 95, 79, 96, 91, 97, 98],
                [100, 101, 110, 102, 118, 103, 119, 104, 120, 111, 121, 122],
            ]
        ]
        node_subset = []
        for group in evolution_groups[0]:
            node_subset.extend(group)
    elif name == "rings.127.E":
        evolution_groups = [
            [
                [4, 5, 15, 6, 22, 7, 23, 8, 24, 16, 25, 26],
                [28, 29, 35, 30, 47, 31, 48, 32, 49, 36, 50, 51],
                [41, 42, 53, 43, 60, 44, 61, 45, 62, 54, 63, 64],
                [66, 67, 73, 68, 85, 69, 86, 70, 87, 74, 88, 89],
            ]
        ]
        node_subset = []
        for group in evolution_groups[0]:
            node_subset.extend(group)
    else:
        raise Exception("Configuration name unknown.")
    if estimation_pairs is None:
        if node_subset is None:
            raise Exception(
                "node_subset must be defined in order to derive the estimation pairs."
            )
        estimation_pairs = partition_qubit_pairs(
            backend, 3, multigraph=False, node_subset=node_subset
        )
    estimation_qubits = partition_qubits(backend, 2, list(np.unique(evolution_groups)))
    all_qubits = partition_qubits(backend, 1, list(np.unique(evolution_groups)))
    return all_qubits, estimation_qubits, estimation_pairs, evolution_groups


def get_backend(s_backend: str):
    from qiskit_ibm_provider import IBMProvider

    provider = IBMProvider(instance="ibm-q/open/main")
    backend = provider.get_backend(s_backend)
    print("Account loaded")
    print("Using " + s_backend)
    return backend
