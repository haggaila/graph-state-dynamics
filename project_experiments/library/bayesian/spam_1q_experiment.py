# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Bayesian estimation of SPAM and gate errors.
"""

from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_experiments.framework import (
    BaseExperiment,
    Options,
    BaseAnalysis,
    AnalysisResultData,
)
from uncertainties import ufloat


class BayesianSPAMExperiment(BaseExperiment):
    """Bayesian estimation experiment for 1Q SPAM and gate errors."""

    def __init__(
        self,
        qubit: int,
        model,
        backend: Optional[Backend] = None,
        add_suffix="",
        b_pulse_gates=False,
    ):
        """Create a new experiment.

        Args:
            qubit: Qubit index on which to run estimation.
            backend: Optional, the backend to run the experiment on.
            add_suffix: An optional suffix string to add to the experiment results.
            b_pulse_gates: Whether to use pulse gates. Important if gate errors are not small
                enough, and in order to meaningfully estimate gate errors.
        """
        super().__init__(
            [qubit],
            analysis=BayesianSPAMAnalysis(add_suffix=add_suffix),
            backend=backend,
        )
        self.set_experiment_options(
            n_x90p_power=model.n_x90p_power,
            gates=model.gates,
            n_repeats=model.n_repeats,
            b_pulse_gates=b_pulse_gates,
        )
        self.analysis.model = model

    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.n_x90p_power = 0
        options.n_repeats = 1
        options.b_pulse_gates = (False,)
        options.gates = []
        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        options = super()._default_transpile_options()
        # Prevents a collapse of all the Rx, Ry gates into one
        options.optimization_level = 0
        return options

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Returns:
            A list of :class:`QuantumCircuit`.

        Raises:
            Exception: In case of unsupported gates requested.
        """
        options = self.experiment_options
        gates = options.gates
        n_x90p_power = options.n_x90p_power
        b_pulse_gates = options.b_pulse_gates
        circuits = []
        pi_2 = np.pi / 2
        n_repeats = options.n_repeats
        r_repeats = range(n_repeats)

        sy_gates = []
        sy_schedules = []
        if b_pulse_gates:
            sx_schedule = self._backend.defaults().instruction_schedule_map.get(
                "sx", self.physical_qubits[0]
            )
            sx = (
                sx_schedule.filter(instruction_types=[pulse.Play])
                .instructions[0][1]
                .pulse
            )
            sy_angles = [-pi_2, pi_2, np.pi]
            sy_names = ["sy", "sydg", "sx2dg"]
            for i_sy_gate, sy_gate_name in enumerate(sy_names):
                sy_gates.append(Gate(sy_gate_name, 1, []))
                with pulse.build() as sy_schedule:
                    pulse.play(
                        pulse.Drag(
                            duration=sx.duration,
                            amp=sx.amp,
                            sigma=sx.sigma,
                            beta=sx.beta,
                            angle=sx.angle + sy_angles[i_sy_gate],
                            name=sy_gate_name,
                        ),
                        pulse.DriveChannel(self.physical_qubits[0]),
                    )
                    sy_schedules.append(sy_schedule)

        for i_gate, s_gate in enumerate(gates):
            circ = QuantumCircuit(1, 1)
            if s_gate == "x":
                circ.x(0)
                circ.barrier()
            elif s_gate[0:4] == "x90p":
                if s_gate == "x90p":
                    n_len = 1
                elif s_gate == "x90p^2":
                    n_len = 2
                elif s_gate == "x90p^5":
                    n_len = 5
                elif s_gate == "x90p^4n":
                    n_len = 4 * n_x90p_power
                elif s_gate == "x90p^(4n+1)":
                    n_len = 4 * n_x90p_power + 1
                else:
                    raise Exception(f"Unknown/unsupported instruction {s_gate}.")
                for _ in range(n_len):
                    circ.sx(0)
                    circ.barrier()
            elif s_gate == "x90m" or s_gate == "x90m^5":
                n_len = 1 if s_gate == "x90m" else 5
                for _ in range(n_len):
                    if b_pulse_gates:
                        circ.append(sy_gates[2], [0])
                    else:
                        circ.sxdg(0)
                    circ.barrier()
                if b_pulse_gates:
                    circ.add_calibration(
                        gate=sy_gates[2],
                        qubits=self.physical_qubits,
                        schedule=sy_schedules[2],
                    )
            elif s_gate == "y90p" or s_gate == "y90p^5":
                n_len = 1 if s_gate == "y90p" else 5
                for _ in range(n_len):
                    if b_pulse_gates:
                        circ.append(sy_gates[0], [0])
                    else:
                        circ.ry(pi_2, 0)
                    circ.barrier()
                if b_pulse_gates:
                    circ.add_calibration(
                        gate=sy_gates[0],
                        qubits=self.physical_qubits,
                        schedule=sy_schedules[0],
                    )
            elif s_gate == "y90m" or s_gate == "y90m^5":
                n_len = 1 if s_gate == "y90m" else 5
                for _ in range(n_len):
                    if b_pulse_gates:
                        circ.append(sy_gates[1], [0])
                    else:
                        circ.ry(-pi_2, 0)
                    circ.barrier()
                if b_pulse_gates:
                    circ.add_calibration(
                        gate=sy_gates[1],
                        qubits=self.physical_qubits,
                        schedule=sy_schedules[1],
                    )
            elif s_gate == "id":
                pass
            else:
                raise Exception(f"Unknown/unsupported instruction {s_gate}.")
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self.experiment_type,
                "qubits": self.physical_qubits,
            }
            if i_gate == 0:
                circ.metadata.update(
                    {
                        "gates": gates,
                        "n_x90p_power": n_x90p_power,
                        "n_repeats": n_repeats,
                    }
                )
            for _ in r_repeats:  # Note that repeats must remain an inner loop
                circuits.append(circ.copy())

        return circuits

    # def _transpiled_circuits(self) -> List[QuantumCircuit]:
    #     """Return a list of experiment circuits, transpiled."""
    #     transpile_opts = copy.copy(self.transpile_options.__dict__)
    #     # transpile_opts["initial_layout"] = list(self.physical_qubits)
    #     # transpile_opts["optimization_level"] = 1
    #     # Transpile level only for this exp.
    #     transpiled = transpile(self.circuits(), self.backend, **transpile_opts)
    #
    #     fig = Figure()
    #     _ = FigureCanvasSVG(fig)
    #     ax = fig.subplots(1, 1, sharex=True)
    #     transpiled[0].draw("mpl", idle_wires=False, ax=ax)
    #     # if self.storage is not None:
    #     #     self.storage["transpiled_circuit_figure"] = fig
    #
    #     # For debugging:
    #     # for i_fig in range(6):
    #     #     fig, ax = plt.subplots(1, 1, sharex=True)
    #     #     transpiled[i_fig].draw("mpl", idle_wires=False, ax=ax)
    #     # plt.show()
    #     return transpiled


class BayesianSPAMAnalysis(BaseAnalysis):
    """Analysis of the Bayesian SPAM experiment."""

    def __init__(self, add_suffix=""):
        super().__init__()
        self.model = None
        self.add_suffix = add_suffix

    def _run_analysis(self, experiment_data):
        # Fetch the probabilities for 00 and 11
        model = self.model
        exp_data = experiment_data.data()
        circ_0 = exp_data[0]
        gates = circ_0["metadata"]["gates"]
        n_x90p_power = circ_0["metadata"]["n_x90p_power"]
        n_repeats = circ_0["metadata"].get("n_repeats", 1)
        if gates != model.gates or n_x90p_power != model.n_x90p_power:
            raise Exception(
                "The gates or X90p power used when creating the circuits differ from the "
                "estimator's settings."
            )
        gate_counts = {}
        i_circuit = 0
        r_repeats = range(n_repeats)
        for i_gate, s_gate in enumerate(gates):
            for i_repeat in r_repeats:  # Note that repeats must remain an inner loop
                gate_data = exp_data[i_circuit]
                counts = gate_data["counts"]
                shots = sum(counts.values())
                cc = counts.get("0", 0.0)
                if i_repeat == 0:
                    gate_counts[(s_gate, i_gate)] = (cc, shots - cc)
                else:
                    gc = gate_counts[(s_gate, i_gate)]
                    gate_counts[(s_gate, i_gate)] = (gc[0] + cc, gc[1] + shots - cc)
                i_circuit += 1

        # result = {'mean': mean, 'cov': cov, 'Var_P': Var_P, 'n_valid_draws': n_valid_draws,
        #           'mean_dict': mean_dict, 'vars_dict': vars_dict}
        result = model.estimate_Bayesian(gate_counts)
        mean_dict = result["mean_dict"]
        vars_dict = result["vars_dict"]
        analysis_results = []
        for key, value in mean_dict.items():
            var = vars_dict.get(key, 0)
            stddev = np.sqrt(var)
            analysis_results.append(
                AnalysisResultData(
                    name=key + self.add_suffix, value=ufloat(value, stddev)
                )
            )
        analysis_results.append(
            AnalysisResultData(
                name="Var_P" + self.add_suffix, value=ufloat(result.get("Var_P", 0), 0)
            )
        )

        return analysis_results, []
