# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import numpy as np
from experiment_routines import run_experiment
from project_experiments.experiment_routines import get_backend

b_short = False
# For debugging, create short sequences (fewer circuits)
b_just_evolution = False
# For debugging, create just the evolution circuits (no estimation)
shots = 1 * 1024
# Number of shots per job

s_backend = "ibm_osaka"
# Choose the relevant backend
edge_groups_name = "rings.127.A"
# E.g., "pairs" "triples" "rings.27" "rings.127.B"
s_init_state = "gr"
# The initial state: "+x" for a product of |+>'s, "gr" for a graph state

n_dd_cycles = [0, 1]
# 0 implies no DD, otherwise the number of sequences between measurements
b_zz_dd = True
# Whether to add staggered DD for the ZZ interactions
b_dilute_evolution = True
# Whether to add a measurements of the DD results at diluted times
b_add_z_measurement = False
# Whether to add Z measurement circuits for all qubits
b_estimation_after_evolution = False
# Whether to insert a second estimation job at the end
b_stabilizers = s_init_state == "gr"
# Whether to measure and simulate the stabilizer dynamics

n_evolution_delays = 11 if b_short else (9 * 4 + 1)
t_final = 5e-6 if b_short else 52e-6
# Final time of the simulation
dt = 1 / 4.5e9
# dt timestep discretization of the device
dt_factor = 4 * 16
# Rounding of the timestep, useful for the DD sequences duration
t_final = (
    np.floor(int(t_final / (n_evolution_delays - 1) / dt) / dt_factor)
    * dt_factor
    * (n_evolution_delays - 1)
    * dt
)

parity_offset_freq = 60e3
# Intended offset frequency shift inserted for improving the signal
parity_delays_cut_off_duration = 70e-6
# Duration of the longest delay in the first part of the parity estimation experiment.
# A Fourier transform will be done on those.
parity_delays_cut_off = 30 if b_short else 50
# Number of measurements in the first part of the parity estimation experiment.
# A Fourier transform will be done on those.
parity_delays_total_duration = (
    180e-6  # Duration of the longest delay in the second part of
)
# the parity estimation experiment, that must be comparable to the T_2 decay times.
parity_delays_extra = 20 if b_short else 25
# Number of measurements in the second part of the parity estimation experiment.

n_zz_delays = 20 if b_short else 30
# Number of delays in the ZZ estimation experiment
delays_zz_ramsey = np.linspace(0, 4e-6, num=n_zz_delays)
# Delays for the ZZ estimation
osc_freq_zz_ramsey = 400e3
# Intended offset frequency shift inserted for improving the signal
evolution_offset_freq = 0
# Offset frequency shift of the qubits evolution experiment. Remains 0.

n_T1_delays = 10
# Number of delays in the T_1 fitting experiment
T1_duration = 500e-6
# Duration of T_1 experiment, should be ~ 2-3 times the best expected T_1's

delays_T1 = np.linspace(0e-6, T1_duration, num=n_T1_delays)
delays_parity = np.append(
    np.linspace(0e-6, parity_delays_cut_off_duration, num=parity_delays_cut_off),
    np.linspace(
        parity_delays_cut_off_duration + 2e-6,
        parity_delays_total_duration,
        num=parity_delays_extra,
    ),
)
delays_evolution = np.linspace(0, t_final, num=n_evolution_delays)  # Must start at 0!

# Resulting number of circuits of the different types. For each job, the number of experiments
# should not pass 300 (to fit in one job), and jobs are grouped as following.
# (1) parity estimation: 2 groups * 2 * (parity_delays_cut_off + parity_delays_extra).
# (2) zz estimation: number of edge groups * 2 * n_zz_delays
# (3) T1 estimation: n_T1_delays
#     SPAM: n_repeats (8 at 1024 shots) * 6
#     evolution: num evolution_groups (1 or 2) * (2 + 1 if Z) * n_evolution_delays *
#               (2 if two dd cycles, +~0.25 if b_dilute_evolution)
# SPAM: n_repeats (8 at 1024 shots) * 6

backend = get_backend(s_backend)
load_jobs = []

run_experiment(
    edge_groups_name,
    parity_offset_freq,
    evolution_offset_freq,
    t_final,
    backend,
    s_init_state,
    delays_T1,
    delays_parity,
    parity_delays_cut_off,
    delays_zz_ramsey,
    osc_freq_zz_ramsey,
    delays_evolution,
    shots=shots,
    b_stabilizers=b_stabilizers,
    n_dd_cycles=n_dd_cycles,
    b_just_evolution=b_just_evolution,
    b_zz_dd=b_zz_dd,
    b_dilute_evolution=b_dilute_evolution,
    b_add_z_measurement=b_add_z_measurement,
    b_estimation_after_evolution=b_estimation_after_evolution,
    load_jobs=load_jobs,
)
