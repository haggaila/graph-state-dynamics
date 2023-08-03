# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import time
import uuid

import pandas as pd

from project_experiments.library.experiment_utils import make_sound
from project_experiments.experiment_routines import get_backend
from project_experiments.library.parity_oscillations import ParityOscillations
from project_experiments.library.zz_ramsey import ZZRamsey
from qiskit_experiments.framework.composite import BatchExperiment, ParallelExperiment
import numpy as np

from project_experiments.load_routines import unpack_analysis_result
from partition import (
    partition_qubits,
    partition_qubit_pairs,
)

s_backend = "ibm_cusco"
backend = get_backend(s_backend)
b_save2file = True
b_save2db = False
b_return_plots = False
n_repeats = 20
shots = 1024
rep_delay = 200 * 1e-6
sleep_seconds = 60

# define which qubits to use
single_qubits_groups = [partition_qubits(backend, distance=2)[0][0:20]]
qubit_pairs_groups = partition_qubit_pairs(backend, distance=3)

# define parameters for the experiments
parity_offset_freq = 60e3
n_T1_delays = 10

delays_T1 = np.linspace(0e-6, 800e-6, num=n_T1_delays)
parity_delays_cut_off = 50
parity_delays_extra = 25
n_zz_delays = 20
delays_parity = np.append(
    np.linspace(0e-6, 70e-6, num=parity_delays_cut_off),
    np.linspace(72e-6, 180e-6, num=parity_delays_extra),
)
delays_zz_ramsey = np.linspace(0e-6, 4e-6, num=n_zz_delays)
osc_freq_zz_ramsey = 400e3

single_qubits_funcs = [
    lambda qubit: ParityOscillations(
        qubit=qubit,
        delays=delays_parity,
        osc_freq=parity_offset_freq,
        cut_off_delay=parity_delays_cut_off,
        fixed_phase=False,
    ),
]
two_qubits_funcs = [
    lambda qubits: ZZRamsey(
        qubit=qubits, delays=delays_zz_ramsey, osc_freq=osc_freq_zz_ramsey
    )
]

analysis_results_list = []
exp_t_list = []
exp_data_list = []
final_exp_list = []
for i_exp in range(n_repeats):
    try:
        par_exps1 = []
        par_exps2 = []
        whole_device_T1 = []
        # --- uncomment for T1 experiment
        # whole_device_T1 = [ParallelExperiment([T1(physical_qubits=[qubit], delays=delays_T1) for qubit in range(backend.num_qubits)])]
        # whole_device_T1[0].analysis.set_options(
        #                 plot=b_return_plots,
        #                 return_data_points=False,
        #                 return_fit_parameters=False,
        #             )
        for group in single_qubits_groups:
            print("Preparing estimation experiments for group: " + str(group))
            for exp_func in single_qubits_funcs:
                exps = []
                for qubit in group:
                    exp = exp_func(qubit[0])
                    exp.analysis.set_options(
                        plot=b_return_plots, return_data_points=False
                    )
                    exps.append(exp)
                par_exp = ParallelExperiment(exps)
                par_exp.set_transpile_options(
                    scheduling_method="alap", optimization_level=0
                )
                par_exps1.append(par_exp)

        # --- uncomment for ZZ experiment
        # for group in qubit_pairs_groups:
        #     print("Preparing estimation experiments for group: " + str(group))
        #     for exp_func in two_qubits_funcs:
        #         exps = []
        #         for qubit_pair in group:
        #             exp = exp_func(qubit_pair)
        #             exp.analysis.set_options(
        #                 plot=b_return_plots,
        #                 return_data_points=False,
        #                 return_fit_parameters=False,
        #             )
        #             exps.append(exp)
        #         par_exp = ParallelExperiment(exps)
        #         par_exp.set_transpile_options(
        #             scheduling_method="alap", optimization_level=0
        #         )
        #         par_exps2.append(par_exp)

        exp = BatchExperiment(
            whole_device_T1 + par_exps1 + par_exps2, flatten_results=True
        )
        print("Executing experiment.")
        exp.set_transpile_options(
            timing_constraints=backend.configuration().timing_constraints,
            scheduling_method="alap",
            optimization_level=0,
        )
        exp_data = exp.run(
            backend=backend, shots=shots, rep_delay=rep_delay, analysis=None
        )
        exp_data_list.append(exp_data)
        final_exp_list.append(exp)
        print(f"Experiment {i_exp + 1} completed. Sleeping {sleep_seconds} seconds.")
        time.sleep(sleep_seconds)

    except Exception as e:
        print("EXCEPTION: " + str(e))

print(f"Finished executing all the experiments")
print(f"Start analysis")

for i_exp in range(n_repeats):
    try:
        exp = final_exp_list[i_exp]
        exp_data = exp_data_list[i_exp]
        exp._set_backend(backend)
        exp._finalize()
        exp_data = exp.analysis.run(exp_data).block_for_results()

        print("Analysis completed.")

        analysis_results = exp_data.analysis_results()
        analysis_results_list.append(analysis_results)
        t = exp_data.jobs()[0].time_per_step().get("finished")
        exp_t_list.append(t)

        if b_save2db:
            # ---- set figure titles -----
            for figkey in range(len(exp_data.figure_names)):
                fig = exp_data.figure(figkey)
                old_title = fig.figure.axes[0].get_title()
                new_title = old_title
                for i, qubit in enumerate(fig.metadata["qubits"]):
                    new_title = f"q{i+1}=Q{qubit}  " + new_title
                fig.figure.axes[0].set_title(new_title)

            exp_data.metadata["rep_delay"] = rep_delay
            exp_data.share_level = "project"
            exp_data.tags = ["graph_state_dynamics"]
            exp_data.save()
            print(f"Experiment {i_exp + 1} saved to cloud.")
    except Exception as e:
        print("EXCEPTION: " + str(e))

print("All experiments saved to cloud.")

# Save to file
if b_save2file:
    data = []
    s_date = exp_t_list[0].strftime("%m_%d_%Y")
    for i, analysis_results in enumerate(analysis_results_list):
        for result in analysis_results:
            data_dict, variables_name = unpack_analysis_result(
                result, s_backend, exp_t_list[i]
            )
            if data_dict is not None:
                data.append(data_dict)

    s_output_path = (
        os.path.abspath("../output") + "/"
    )  # use the absolute path of the current file
    if not os.path.exists(s_output_path):
        os.mkdir(s_output_path)
    s_time_series_path = s_output_path + "time_series/"
    if not os.path.exists(s_time_series_path):
        os.mkdir(s_time_series_path)
    S_DATA_PREFIX = "Time_Series"

    df = pd.DataFrame(data)
    s_uuid = uuid.uuid4().hex
    s_file_path = (
        s_time_series_path
        + s_backend
        + "."
        + S_DATA_PREFIX
        + "."
        + s_date
        + "."
        + s_uuid
    )
    df.to_csv(s_file_path + ".csv", index=False)
    print("Saved to file")
    make_sound()
