# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from typing import Sequence, Tuple
import copy
import numpy as np
from qiskit import transpile, QuantumCircuit, circuit, pulse
from qiskit.circuit.library import XGate
from qiskit.providers.backend import Backend
from qiskit.test.mock import FakeBackend
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.passes import (
    ALAPSchedule,
    DynamicalDecoupling,
    PadDynamicalDecoupling,
    ConstrainedReschedule,
    TimeUnitConversion,
    ALAPScheduleAnalysis,
    PadDelay,
)

from qiskit_experiments.framework import BaseExperiment, Options
from typing import List, Optional
import lmfit
from qiskit.circuit import Gate
from project_experiments.library.experiment_utils import myfft, combine_curve_data
from qiskit_experiments.data_processing import (
    DataProcessor,
    Probability,
    MarginalizeCounts,
)
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis import CurveData
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure


class MultiqubitXY(BaseExperiment):
    def __init__(
        self,
        qubits: Sequence[int],
        delays: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None,
        osc_freq: float = 0.0,
        cut_off_delay: Optional[int] = None,
        file_path_prefix="",
        s_init_state="+x",
        pair_groups=Sequence[Sequence[int]],
        storage: Optional[dict] = None,
        b_stabilizers=False,
        topology_index=Optional[Sequence[int]],
        n_dd_cycles: Optional[int] = 0,
        b_zz_dd=False,
        zz_dd_group_indexes=None,
        b_add_z_measurement=False,
    ):
        """Create new experiment.
        Args:
            qubit: The qubits on which to run the Ramsey XY experiment.
            backend: Optional, the backend to run the experiment on.
            delays: The delays to scan, in seconds
        """
        super().__init__(physical_qubits=qubits, backend=backend)
        self.analysis = MultiqubitXYAnalysis(
            physical_qubits=self.physical_qubits,
            osc_freq=osc_freq,
            cut_off_delay=cut_off_delay,
            file_path_prefix=file_path_prefix,
            storage=storage,
            b_stabilizers=b_stabilizers,
            topology_index=topology_index,
            b_add_z_measurement=b_add_z_measurement,
        )
        if s_init_state != "+x" and s_init_state != "gr":
            raise Exception(f"Unsupported init_state argument: {s_init_state}.")
        self.s_init_state = s_init_state
        self.pair_groups = pair_groups
        self.granularity = 1
        self.set_experiment_options(osc_freq=osc_freq)
        if delays is not None:
            self.set_experiment_options(delays=delays)
        self.storage = storage
        self.b_stabilizers = b_stabilizers
        self.b_add_z_measurement = b_add_z_measurement
        self.topo_index = topology_index
        self.n_dd_cycles = n_dd_cycles
        self.b_zz_dd = b_zz_dd
        self.zz_dd_group_indexes = zz_dd_group_indexes

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the ZZ Ramsey experiment.
        Experiment Options:
            delays (list): The list of delays that will be scanned in the experiment, in seconds.
            zz_rotations (float): number of full rotations of the Bloch vector if ZZ is zero
        """
        options = super()._default_experiment_options()
        options.delays = np.linspace(0e-6, 50e-6, 51)
        options.osc_freq = 0.0

        return options

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)

        timing_constraints = getattr(self.transpile_options, "timing_constraints", {})
        if "granularity" in timing_constraints:
            self.granularity = timing_constraints["granularity"]

        # Scheduling parameters
        if not self._backend.configuration().simulator and not isinstance(
            backend, FakeBackend
        ):
            if "acquire_alignment" not in timing_constraints:
                timing_constraints["acquire_alignment"] = 16
            scheduling_method = getattr(
                self.transpile_options, "scheduling_method", "alap"
            )
            self.set_transpile_options(
                timing_constraints=timing_constraints,
                scheduling_method=scheduling_method,
            )

    def circuits(self) -> List[circuit.QuantumCircuit]:
        """Create the circuits for the ZZ Ramsey characterization experiment.
        Returns:
            A list of circuits with a variable delay.
        """

        metadata = {
            "experiment_type": self._type,
            "init_state": self.s_init_state,
            "qubits": self.physical_qubits,
            "unit": "s",
        }
        config = self.backend.configuration()
        if hasattr(config, "dt"):
            dt_unit = True
            dt_factor = config.dt
        else:
            dt_unit = False

        circs = []
        n = self.num_qubits
        qubits = [i for i in range(n)]
        properties = self.backend.properties()
        physical_qubits = self.physical_qubits
        physical_groups = []
        gate_len_groups = []
        for group in self.pair_groups:
            physical_group = []
            gate_len_group = []
            for qubit_pair in group:
                i = physical_qubits.index(qubit_pair[0])
                j = physical_qubits.index(qubit_pair[1])
                inverted_pair = [qubit_pair[1], qubit_pair[0]]
                original_gate_len = None
                inverted_gate_len = None
                s_gate = "cx"
                try:
                    original_gate_len = properties.gate_length(s_gate, qubit_pair)
                except Exception as ex:
                    s_gate = "ecr"
                    try:
                        original_gate_len = properties.gate_length(s_gate, qubit_pair)
                    except Exception as ex:
                        pass
                try:
                    inverted_gate_len = properties.gate_length(s_gate, inverted_pair)
                except Exception as ex:
                    pass
                if inverted_gate_len is None and original_gate_len is None:
                    raise Exception(
                        f"Gate could not be determined for pair {qubit_pair}."
                    )
                if inverted_gate_len is None or inverted_gate_len > original_gate_len:
                    physical_pair = [i, j]
                    gate_len = original_gate_len + 2 * properties.gate_length(
                        "sx", qubit_pair[1]
                    )
                else:
                    physical_pair = [j, i]
                    gate_len = inverted_gate_len + 2 * properties.gate_length(
                        "sx", qubit_pair[0]
                    )
                physical_group.append(physical_pair)
                gate_len_group.append(gate_len)
            physical_groups.append(physical_group)
            gate_len_groups.append(gate_len_group)
        if self.storage is not None:
            self.storage["gate_len_groups"] = gate_len_groups
        x_gate_len = properties.gate_length("x", 0)
        # gate length of exemplary qubit, 0. Will work best on devices where lengths are equal

        n_dd_cycles = self.n_dd_cycles
        b_zz_dd = self.b_zz_dd
        if b_zz_dd:
            n_delays_per_cycle = 4
        else:
            n_delays_per_cycle = 2
        r_dd_cycles = range(n_delays_per_cycle * self.n_dd_cycles)
        delays = self.experiment_options.delays
        dd_delays = []
        dd_group_qubits = ([], [])
        if b_zz_dd:
            for i_qubit, i_group in enumerate(self.zz_dd_group_indexes):
                dd_group_qubits[i_group].append(qubits[i_qubit])
        for i_delay, delay in enumerate(self.experiment_options.delays):
            if dt_unit:
                delay_dt = round(delay / dt_factor)
                real_delay_in_sec = delay_dt * dt_factor
            else:
                real_delay_in_sec = delay

            rotation_angle = (
                2 * np.pi * self.experiment_options.osc_freq * real_delay_in_sec
            )

            circ = QuantumCircuit(n, n)

            # circ.barrier()
            circ.sx(qubits)
            circ.rz(np.pi / 2, qubits)
            circ.barrier()

            if self.s_init_state == "gr":
                # for i in range(n):  # Old version, based on direct coupling map
                #     for j in range(i + 1, n):
                #         if [physical_qubits[i], physical_qubits[j]] in coupling_map:
                #             circ.cz(i, j)
                for i_group, cz_group in enumerate(physical_groups):
                    for physical_pair in cz_group:
                        circ.cz(physical_pair[0], physical_pair[1])
                    circ.barrier()
                    # for j_group, x_group in enumerate(physical_groups):
                    #     if i_group == j_group:
                    #         pass
                    #     for physical_pair in x_group:
                    #         pass
            if n_dd_cycles == 0:
                if dt_unit:
                    circ.delay(delay_dt, qubits, "dt")
                else:
                    circ.delay(delay, qubits, "s")
                circ.barrier()
            else:
                if i_delay == 0:
                    if delay > 0.0:
                        raise Exception(
                            "When DD is requested, the first delay must be 0."
                        )
                else:
                    # x_dagger = Gate('x_dagger', 1, params=[])
                    # circ = self._x_dagger_gates_calibration(x_dagger, circ=circ)
                    delay_interval = delay - delays[i_delay - 1]
                    dd_delay = (
                        delay_interval - n_delays_per_cycle * n_dd_cycles * x_gate_len
                    ) / n_delays_per_cycle
                    if dd_delay < 0.0:
                        raise Exception(
                            f"The {i_delay}'th delay interval is too short to fit "
                            f"{n_dd_cycles} CPMG cycles."
                        )
                    for _ in r_dd_cycles:
                        dd_delays.append(dd_delay)
                    for i_dd_delay, dd_delay in enumerate(dd_delays):
                        if b_zz_dd:
                            x_qubits = dd_group_qubits[np.mod(i_dd_delay, 2)]
                        else:
                            x_qubits = qubits
                        if dt_unit:
                            circ.delay(round(dd_delay / dt_factor), qubits, "dt")
                        else:
                            circ.delay(dd_delay, qubits, "s")
                        circ.barrier()
                        _gate = "x"
                        if b_zz_dd:  # 0,1 are X gates, 2,3 are X dagger
                            if np.mod(i_dd_delay, 4) >= 2:
                                _gate = "x_dagger"
                        else:  # X gates then X dagger
                            if np.mod(i_dd_delay, 2):
                                _gate = "x_dagger"
                        if _gate == "x_dagger":
                            circ.rz(np.pi, x_qubits)
                            circ.barrier()
                            circ.x(x_qubits)
                            circ.barrier()
                            circ.rz(np.pi, x_qubits)
                            circ.barrier()
                            # for q in x_qubits:
                            #     circ.append(x_dagger, [q])
                        else:
                            circ.x(x_qubits)
                        # circ.x(x_qubits)
                        circ.barrier()

            if rotation_angle != 0.0:
                circ.rz(rotation_angle, qubits)
                # circ.barrier()

            if self.b_stabilizers:
                circ_x = circ.copy()
                circ_y = circ.copy()

                # ZXZX , measure X for odd qubits on the ring
                for i in range(1, len(self.topo_index), 2):
                    circ_x.rz(-np.pi / 2, self.topo_index[i])
                    circ_x.sxdg(self.topo_index[i])
                circ_x.barrier()
                circ_x.measure(qubits, qubits)

                circ_x.metadata = metadata.copy()
                circ_x.metadata["series"] = "X"
                circ_x.metadata["xval"] = real_delay_in_sec

                # XZXZ , measure X for even qubits on the ring
                for i in range(0, len(self.topo_index), 2):
                    circ_y.rz(-np.pi / 2, self.topo_index[i])
                    circ_y.sxdg(self.topo_index[i])
                circ_y.barrier()
                circ_y.measure(qubits, qubits)

                circ_y.metadata = metadata.copy()
                circ_y.metadata["series"] = "Y"
                circ_y.metadata["xval"] = real_delay_in_sec

                circs.extend([circ_x, circ_y])
            else:
                circ_x = circ.copy()
                circ_y = circ.copy()

                circ_x.rz(-np.pi / 2, qubits)
                circ_x.sxdg(qubits)
                circ_x.barrier()
                circ_x.measure(qubits, qubits)

                circ_x.metadata = metadata.copy()
                circ_x.metadata["series"] = "X"
                circ_x.metadata["xval"] = real_delay_in_sec

                circ_y.rz(np.pi, qubits)
                circ_y.sxdg(qubits)
                circ_y.barrier()
                circ_y.measure(qubits, qubits)

                circ_y.metadata = metadata.copy()
                circ_y.metadata["series"] = "Y"
                circ_y.metadata["xval"] = real_delay_in_sec

                if self.b_add_z_measurement:
                    circ_z = circ.copy()
                    circ_z.measure(qubits, qubits)

                    circ_z.metadata = metadata.copy()
                    circ_z.metadata["series"] = "Z"
                    circ_z.metadata["xval"] = real_delay_in_sec

                    circs.extend([circ_x, circ_y, circ_z])
                else:
                    circs.extend([circ_x, circ_y])
        return circs

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled.

        This function can be overridden to define custom transpilation.
        """
        transpile_opts = copy.copy(self.transpile_options.__dict__)
        transpile_opts["initial_layout"] = list(self.physical_qubits)
        transpile_opts[
            "optimization_level"
        ] = 1  # Extra transpile level only for this exp.
        transpiled = transpile(self.circuits(), self.backend, **transpile_opts)

        # durations = InstructionDurations.from_backend(self.backend)
        # dd_sequence = [XGate(), XGate()]
        # pm = PassManager([ALAPScheduleAnalysis(durations),
        #                   ConstrainedReschedule(16, 16),
        #                   # PadDynamicalDecoupling(durations, dd_sequence, pulse_alignment=16)
        #                   ])
        # transpiled = pm.run(transpiled)

        fig = Figure()
        _ = FigureCanvasSVG(fig)
        ax = fig.subplots(1, 1, sharex=True)
        transpiled[3].draw("mpl", idle_wires=False, ax=ax)
        if self.storage is not None:
            self.storage["transpiled_circuit_figure"] = fig
        return transpiled

    def _x_dagger_gates_calibration(self, x_dagger: Gate, circ: QuantumCircuit):
        for q in range(self.num_qubits):
            x_pulse = (
                self.backend.defaults()
                .instruction_schedule_map.get("x", self.physical_qubits[q])
                .instructions[0][1]
            )
            with pulse.build(self.backend, name="x_dagger") as x_dagger_schedule:
                pulse.play(
                    pulse.Drag(
                        duration=x_pulse.pulse.duration,
                        amp=x_pulse.pulse.amp * -1,
                        sigma=x_pulse.pulse.sigma,
                        beta=x_pulse.pulse.beta * 1,
                        angle=x_pulse.pulse.angle,
                    ),
                    channel=x_pulse.channel,
                )
            circ.add_calibration(
                x_dagger, [self.physical_qubits[q]], schedule=x_dagger_schedule
            )

        return circ


class MultiqubitXYAnalysis(curve.CurveAnalysis):
    def __init__(
        self,
        physical_qubits: Tuple[int],
        osc_freq: float = 0,
        cut_off_delay: Optional[int] = None,
        file_path_prefix="",
        storage: Optional[dict] = None,
        b_stabilizers=False,
        topology_index=Optional[Sequence[int]],
        b_add_z_measurement=False,
    ):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="a*x",
                    name="_",
                    data_sort_key={"series": "_"},
                ),
            ]
        )
        self.storage = storage

        self.physical_qubits = physical_qubits
        self._osc_freq = osc_freq
        self._cut_off_delay = cut_off_delay
        self.b_add_z_measurement = b_add_z_measurement
        self.b_stabilizers = b_stabilizers
        self.topo_index = topology_index
        self._file_path_prefix = file_path_prefix
        self.measurements = ("X", "Y", "Z") if self.b_add_z_measurement else ("X", "Y")
        # fake model to get the data from circuits
        self._auxiliary_models = [
            lmfit.models.ExpressionModel(
                expr="a*x",
                name=measurement,
            )
            for measurement in self.measurements
        ]
        self._options.data_subfit_map = {
            measurement: {"series": measurement} for measurement in self.measurements
        }
        self._plot_fourier = False
        self._save_data = True if self._file_path_prefix != "" else False

    def _run_analysis(self, experiment_data):
        figures = []
        if self.storage is not None:
            fig = self.storage.get("transpiled_circuit_figure", None)
            if fig is not None:
                figures.append(fig)

        if self.b_stabilizers:
            data3q = self._stabilizer_data(experiment_data)

            n = len(self.topo_index)

            fig = Figure(figsize=(6.4, 2.5 * n))
            _ = FigureCanvasSVG(fig)
            axs = fig.subplots(n, 1, sharex=True)

            for i in range(n):
                outcome = "000"
                stabilizer = data3q.get_subset_of(
                    f"stabilizer_{self.topo_index[i]}_P{outcome}"
                )
                axs[i].errorbar(
                    stabilizer.x * 1e6,
                    stabilizer.y,
                    yerr=stabilizer.y_err,
                    fmt="bo",
                    alpha=0.5,
                    capsize=4,
                    markersize=5,
                    label=f"XZZ q_topo_{self.topo_index[i]},q_{i} ",
                )
                axs[i].set_ylabel(r"$P(000)$")
                axs[i].legend(loc="upper right", frameon=False)
            axs[n - 1].set_xlabel(r"$Delay [\mu s]$")
            axs[0].set_title(f"Driven freq = {round(self._osc_freq / 1e3, 1)}kHz")

            figures.append(fig)

            # save single qubit X data
            data_1q = self._stabilizers_1q_data(experiment_data)

        else:
            data = self._single_qubits_data(experiment_data)

            qubits = self.physical_qubits
            n = len(self.physical_qubits)

            fig = Figure(figsize=(6.4, 2.5 * n))
            _ = FigureCanvasSVG(fig)
            axs = fig.subplots(n, 1, sharex=True)
            for i, q in enumerate(qubits):
                RamX, RamY = data.get_subset_of(f"X_{q}"), data.get_subset_of(f"Y_{q}")
                axs[i].errorbar(
                    RamX.x * 1e6,
                    RamX.y,
                    yerr=RamX.y_err,
                    fmt="bo",
                    alpha=0.5,
                    capsize=4,
                    markersize=5,
                    label=f"X q_{q}",
                )
                axs[i].errorbar(
                    RamY.x * 1e6,
                    RamY.y,
                    yerr=RamY.y_err,
                    fmt="go",
                    alpha=0.5,
                    capsize=4,
                    markersize=5,
                    label=f"Y q_{q}",
                )
                if self.b_add_z_measurement:
                    RamZ = data.get_subset_of(f"Z_{q}")
                    axs[i].errorbar(
                        RamZ.x * 1e6,
                        RamZ.y,
                        yerr=RamZ.y_err,
                        fmt="ro",
                        alpha=0.5,
                        capsize=4,
                        markersize=5,
                        label=f"Z q_{q}",
                    )
                axs[i].set_ylabel(r"$P(0)$")
                axs[i].legend(loc="upper right", frameon=False)
            axs[n - 1].set_xlabel(r"$Delay [\mu s]$")
            axs[0].set_title(f"Driven freq = {round(self._osc_freq / 1e3, 1)}kHz")

            figures.append(fig)

            # ---- saving data for correlations
            self._2qubits_prob(experiment_data)

            # m = int(n*(n-1)/2)
            # fig2 = Figure(figsize=(6.4, 2.5 * m))
            # _ = FigureCanvasSVG(fig2)
            # axs2 = fig2.subplots(m, 1, sharex=True)
            # corr = self._correlations(experiment_data=experiment_data)
            # c = 0
            # for i in range(n):
            #     for j in range(i + 1, n):
            #         XX = corr.get_subset_of(f'X_{qubits[i]}_X_{qubits[j]}')
            #         axs2[c].errorbar(XX.x * 1e6, XX.y, yerr=XX.y_err, fmt='bo',
            #                         alpha=0.5, capsize=4, markersize=5, label=f'X_{qubits[i]}_X_{qubits[j]}')
            #         axs2[c].set_ylabel(r'$P(0)$')
            #         axs2[c].legend(loc='upper right', frameon=False)
            #         c += 1
            # axs2[m - 1].set_xlabel(r'$Delay [\mu s]$')
            #
            #
            # figures.append(fig2)

            if self._plot_fourier:
                figures.append(self._fourier_fig(data=data))

        return [], figures

    def _stabilizer_data(self, experiment_data) -> CurveData:
        raw_data = experiment_data.data()
        formatted_data_list = []
        labels = []
        n_qubits = len(self.topo_index)
        for i in range(n_qubits):
            for outcome in ("000", "001", "010", "011", "100", "101", "110", "111"):
                self._options.data_processor = DataProcessor(
                    input_key="counts",
                    data_actions=[
                        MarginalizeCounts(
                            {
                                self.topo_index[np.mod(i, n_qubits)],
                                self.topo_index[np.mod(i - 1, n_qubits)],
                                self.topo_index[np.mod(i + 1, n_qubits)],
                            }
                        ),
                        Probability(outcome=outcome),
                    ],
                )
                # Run data processing
                processed_data = self._run_data_processing(
                    raw_data=raw_data,
                    models=self._auxiliary_models,
                )

                if np.mod(i, 2) == 0:
                    subset = "Y"
                else:
                    subset = "X"

                # Format data
                formatted_data_list.append(
                    self._format_data(processed_data.get_subset_of(subset))
                )
                labels.append(f"stabilizer_{self.topo_index[i]}_P{outcome}")

        formatted_data = combine_curve_data(formatted_data_list, labels)

        if self._save_data:
            import pickle

            with open(self._file_path_prefix + ".3Q.pkl", "wb") as handle:
                pickle.dump(formatted_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return formatted_data

    def _stabilizers_1q_data(self, experiment_data) -> CurveData:
        formatted_data_list = []
        n_qubits = len(self.physical_qubits)
        labels = []
        for i in range(n_qubits):
            self._options.data_processor = DataProcessor(
                input_key="counts",
                data_actions=[
                    MarginalizeCounts({self.topo_index[i]}),
                    Probability(outcome="0"),
                ],
            )
            # Run data processing
            processed_data = self._run_data_processing(
                raw_data=experiment_data.data(),
                models=self._auxiliary_models,
            )

            if np.mod(i, 2) == 0:
                subset = "Y"
            else:
                subset = "X"

            # Format data
            formatted_data_list.append(
                self._format_data(processed_data.get_subset_of(subset))
            )
            labels.append(f"X_{self.topo_index[i]}")

            if np.mod(i, 2) == 0:
                subset = "X"
            else:
                subset = "Y"

            # Format data
            formatted_data_list.append(
                self._format_data(processed_data.get_subset_of(subset))
            )
            labels.append(f"Z_{self.topo_index[i]}")

        formatted_data = combine_curve_data(formatted_data_list, labels)

        if self._save_data:
            import pickle

            with open(self._file_path_prefix + ".pkl", "wb") as handle:
                pickle.dump(formatted_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return formatted_data

    def _single_qubits_data(self, experiment_data) -> CurveData:
        self._labels = [
            f"{base}_{q}" for q in self.physical_qubits for base in self.measurements
        ]
        formatted_data_list = []
        n_qubits = len(self.physical_qubits)
        for i in range(n_qubits):
            self._options.data_processor = DataProcessor(
                input_key="counts",
                data_actions=[MarginalizeCounts({i}), Probability(outcome="0")],
            )
            # Run data processing
            processed_data = self._run_data_processing(
                raw_data=experiment_data.data(),
                models=self._auxiliary_models,
            )
            # Format data
            for subset in self.measurements:
                formatted_data_list.append(
                    self._format_data(processed_data.get_subset_of(subset))
                )

        formatted_data = combine_curve_data(formatted_data_list, self._labels)

        if self._save_data:
            import pickle

            with open(self._file_path_prefix + ".pkl", "wb") as handle:
                pickle.dump(formatted_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return formatted_data

    def _2qubits_prob(self, experiment_data) -> CurveData:
        raw_data = experiment_data.data()
        formatted_data_list = []
        labels = []
        n_qubits = len(self.physical_qubits)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                for outcome in ("00", "01", "10", "11"):
                    self._options.data_processor = DataProcessor(
                        input_key="counts",
                        data_actions=[
                            MarginalizeCounts({i, j}),
                            Probability(outcome=outcome),
                        ],
                    )
                    # Run data processing
                    processed_data = self._run_data_processing(
                        raw_data=raw_data,
                        models=self._auxiliary_models,
                    )

                    for subset in self.measurements:
                        # the subset is the name of the circuit, to modify to XY change also the names of the new labels

                        # Format data
                        formatted_data_list.append(
                            self._format_data(processed_data.get_subset_of(subset))
                        )
                        labels.append(
                            f"{subset}_{self.physical_qubits[i]}_{subset}_{self.physical_qubits[j]}_P{outcome}"
                        )

        formatted_data = combine_curve_data(formatted_data_list, labels)

        if self._save_data:
            import pickle

            with open(self._file_path_prefix + ".2Q.pkl", "wb") as handle:
                pickle.dump(formatted_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return formatted_data

    def _correlations(self, experiment_data) -> CurveData:
        expectation_values = self._single_qubits_data(experiment_data)
        ev = expectation_values
        raw_data = experiment_data.data()
        formatted_data_list = []
        labels = []
        n_qubits = len(self.physical_qubits)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                self._options.data_processor = DataProcessor(
                    input_key="counts",
                    data_actions=[MarginalizeCounts({i, j}), Probability(outcome="00")],
                )
                # Run data processing
                processed_data_00 = self._run_data_processing(
                    raw_data=raw_data,
                    models=self._auxiliary_models,
                ).get_subset_of("X")
                self._options.data_processor = DataProcessor(
                    input_key="counts",
                    data_actions=[MarginalizeCounts({i, j}), Probability(outcome="11")],
                )
                # Run data processing
                processed_data_11 = self._run_data_processing(
                    raw_data=raw_data,
                    models=self._auxiliary_models,
                ).get_subset_of("X")

                ev_Xi = ev.get_subset_of(f"X_{self.physical_qubits[i]}")
                ev_Xj = ev.get_subset_of(f"X_{self.physical_qubits[j]}")
                y = (
                    2 * (processed_data_00.y + processed_data_11.y - 0.5)
                    - ev_Xi.y * ev_Xj.y
                )
                # y_err = processed_data_00.y + processed_data_11.y
                processed_data = CurveData(
                    x=processed_data_11.x,
                    y=y,
                    y_err=processed_data_11.y_err * 0,
                    shots=processed_data_11.shots,
                    data_allocation=processed_data_11.data_allocation,
                    labels=processed_data_11.labels,
                )
                # Format data
                formatted_data_list.append(self._format_data(processed_data))
                labels.append(
                    f"X_{self.physical_qubits[i]}_X_{self.physical_qubits[j]}"
                )

        formatted_data = combine_curve_data(formatted_data_list, labels)

        return formatted_data

    def _fourier_data(self, data: CurveData):
        fourier_data = dict()
        for series in data.labels:
            sub_data = data.get_subset_of(series)
            if self._cut_off_delay is not None:
                fourier_signal, freq_vec = myfft(
                    sub_data.y[: self._cut_off_delay]
                    - np.mean(sub_data.y[: self._cut_off_delay]),
                    dt=(sub_data.x[1] - sub_data.x[0]),
                )
            else:
                fourier_signal, freq_vec = myfft(
                    sub_data.y - np.mean(sub_data.y), dt=(sub_data.x[1] - sub_data.x[0])
                )
            fourier_data[series] = {
                "fourier_signal": fourier_signal,
                "freq_vec": freq_vec,
            }

        return fourier_data

    def _fourier_fig(self, data: CurveData):
        fourier_data = self._fourier_data(data=data)
        fig_fourier = Figure()
        _ = FigureCanvasSVG(fig_fourier)
        ax = fig_fourier.subplots(1, 1)
        for series in self._labels:
            ax.plot(
                fourier_data[series]["freq_vec"] / 1e3,
                abs(fourier_data[series]["fourier_signal"]),
                label=series,
            )
            ax.set_xlim([0, max(fourier_data[series]["freq_vec"] / 1e3)])
        ax.set_xlabel("frequency [kHz]")
        ax.set_ylabel("abs(FFT)")
        ax.legend()
        return fig_fourier
