# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import uuid
from project_experiments.experiment_routines import get_backend
from project_experiments.library.parity_oscillations import ParityOscillations
from qiskit_experiments.framework.composite import BatchExperiment, ParallelExperiment
import numpy as np
from project_experiments.partition import partition_qubits

b_save_experiments = False
"""Set to true in order to save qiskit-experiments to a database service"""

s_backend = "ibm_cusco"
backend = get_backend(s_backend)

shots = 1024

qubit_groups = partition_qubits(backend, distance=2)
load_jobs = []  # insert job id to analyse the same job's results again

# define parameters
cut_off_delay = 120
delays_p = np.linspace(0e-6, 250e-6, num=cut_off_delay)
osc_freq = 50e3

s_output_path = (
    os.path.abspath("../output") + "/"
)  # use the absolute path of the current file
if not os.path.exists(s_output_path):
    os.mkdir(s_output_path)
s_time_series_path = s_output_path + "PO/"
if not os.path.exists(s_time_series_path):
    os.mkdir(s_time_series_path)
S_DATA_PREFIX = "nu"
s_uuid = uuid.uuid4().hex
s_file_path = s_time_series_path + s_backend + "." + S_DATA_PREFIX + "." + s_uuid

# uncomment depend on which fitter to use
estimation_funcs = [
    lambda qubit: ParityOscillations(
        qubit=qubit,
        delays=delays_p,
        osc_freq=osc_freq,
        cut_off_delay=cut_off_delay,
        fixed_phase=False,
        file_path_prefix=s_file_path,
        # fixed_b=False,
        # b_kappa=True,
    ),
]

par_exps1 = []
for group in qubit_groups:
    print("Preparing estimation experiments for group: " + str(group))
    for exp_func in estimation_funcs:
        exps = []
        for qubit in group:
            exp = exp_func(qubit[0])
            exps.append(exp)
        par_exp = ParallelExperiment(exps)
        par_exp.set_transpile_options(scheduling_method="alap", optimization_level=0)
        par_exps1.append(par_exp)

exp = BatchExperiment(par_exps1, flatten_results=True)
print("Executing experiment.")
exp.set_transpile_options(
    timing_constraints=backend.configuration().timing_constraints,
    scheduling_method="alap",
    optimization_level=0,
)

if load_jobs:
    exp._set_backend(backend)
    exp._finalize()
    expdata = exp._initialize_experiment_data()
    print("Loading jobs")
    retrieved_jobs = []
    for job_id in load_jobs:
        retrieved_job = backend.provider.backend.retrieve_job(job_id)
        retrieved_jobs.append(retrieved_job)
    expdata.add_jobs(retrieved_jobs)
    print("Finished loading jobs")
    exp_data = exp.analysis.run(expdata)
else:
    exp_data = exp.run(backend=backend, shots=shots)

exp_data.block_for_results()

print("Experiment completed.")

analysis_results = exp_data.analysis_results()

# ---- set figure titles -----
for figkey in range(len(exp_data.figure_names)):
    fig = exp_data.figure(figkey)
    old_title = fig.figure.axes[0].get_title()
    new_title = old_title
    for i, qubit in enumerate(fig.metadata["qubits"]):
        new_title = f"q{i+1}=Q{qubit}  " + new_title
    fig.figure.axes[0].set_title(new_title)

if b_save_experiments:
    exp_data.share_level = "project"
    exp_data.tags = ["graph_state_dynamics"]
    exp_data.save()
