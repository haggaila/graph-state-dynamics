# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import List, Optional, Sequence
import numpy as np
from qiskit import QuantumCircuit, circuit
from qiskit.providers.backend import Backend
from qiskit.test.mock import FakeBackend
from qiskit.circuit import Parameter

from project_experiments.library.zz_ramsey_analysis import ZZRamseyAnalysis
from qiskit_experiments.framework import BaseExperiment, Options


class ZZRamsey(BaseExperiment):
    r"""Experiment to characterize ZZ interaction between qubits
    # section: Overview
        ZZ is a measure of the extent to which the state of one qubit can affect the state of some
        other qubit. It is the difference in the qubit frequency that occurs when a different qubit is in the |0> state
        vs. the |1> state. Experimentally, we measure  ZZ by employing modified Ramsey sequences on Q0 while Q1 is in
        either the |0〉 or |1〉state. The Ramsey sequence is modified by including an echo pulse and an Rz pulse. The echo
        pulse allows us to cancel some undesirable effects that would otherwise cause errors to accumulate. The Rz pulse
        adds a faster rotation to the measured qubit state to help distinguish between slow frequency offsets and qubit
        decay. This is because it is easier to fit a fast sinusoid than a very slow one.
        This experiment consists of following two circuits:
        .. parsed-literal::
                 ┌────┐ ░ ┌─────────────────┐ ░ ┌───┐ ░ ┌─────────────────┐ ░ ┌─────────────────────┐┌────┐ ░ ┌─┐
            q_0: ┤ √X ├─░─┤ Delay(delay[s]) ├─░─┤ X ├─░─┤ Delay(delay[s]) ├─░─┤ Rz(2*delay*dt*f*pi) ├┤ √X ├─░─┤M├
                 └────┘ ░ └─────────────────┘ ░ ├───┤ ░ └─────────────────┘ ░ └────────┬───┬────────┘└────┘ ░ └╥┘
            q_1: ───────░─────────────────────░─┤ X ├─░─────────────────────░──────────┤ X ├────────────────░──╫─
                        ░                     ░ └───┘ ░                     ░          └───┘                ░  ║
            c: 1/══════════════════════════════════════════════════════════════════════════════════════════════╩═
                                                                                                               0
            Ramsey sequence applied to target qubit q_0 with control qubit q_1 prepared in |0> state
                 ┌────┐ ░ ┌─────────────────┐ ░ ┌───┐ ░ ┌─────────────────┐ ░ ┌─────────────────────┐┌────┐ ░ ┌─┐
            q_0: ┤ √X ├─░─┤ Delay(delay[s]) ├─░─┤ X ├─░─┤ Delay(delay[s]) ├─░─┤ Rz(2*delay*dt*f*pi) ├┤ √X ├─░─┤M├
                 ├───┬┘ ░ └─────────────────┘ ░ ├───┤ ░ └─────────────────┘ ░ └─────────────────────┘└────┘ ░ └╥┘
            q_1: ┤ X ├──░─────────────────────░─┤ X ├─░─────────────────────░───────────────────────────────░──╫─
                 └───┘  ░                     ░ └───┘ ░                     ░                               ░  ║
            c: 1/══════════════════════════════════════════════════════════════════════════════════════════════╩═
            Ramsey sequence applied to target qubit q_0 with control qubit q_1 prepared in |1> state                                                                                                   0
        The first and second circuits measure the expectation value along the -Y axis when the control qubit is prepared in |0〉 or |1〉 respectively.
    # section: analysis_ref
        :py:class:`ZZRamseyAnalysis`
    """

    def __init__(
        self,
        qubit: (int, int),
        delays: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None,
        osc_freq: float = 0.0,
    ):
        """Create new experiment.
        Args:
            qubit: The qubits on which to run the Ramsey XY experiment.
            backend: Optional, the backend to run the experiment on.
            delays: The delays to scan, in seconds
        """
        super().__init__(
            physical_qubits=qubit, analysis=ZZRamseyAnalysis(), backend=backend
        )
        self.analysis.set_options(outcome="1")
        self.granularity = 1
        if delays is not None:
            self.set_experiment_options(delays=delays, osc_freq=osc_freq)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the ZZ Ramsey experiment.
        Experiment Options:
            delays (list): The list of delays that will be scanned in the experiment, in seconds.
            zz_rotations (float): number of full rotations of the Bloch vector if ZZ is zero
        """
        options = super()._default_experiment_options()
        options.delays = np.linspace(1e-6, 10e-6, 101)
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
            "qubits": self.physical_qubits,
            "unit": "s",
        }

        if self.backend and hasattr(self.backend.configuration(), "dt"):
            dt_unit = True
            dt_factor = self.backend.configuration().dt
        else:
            dt_unit = False

        circs = []
        for delay in self.experiment_options.delays:
            if dt_unit:
                delay_dt = round(delay / dt_factor)
                real_delay_in_sec = delay_dt * dt_factor
            else:
                real_delay_in_sec = delay

            rotation_angle = (
                2 * np.pi * self.experiment_options.osc_freq * real_delay_in_sec
            )

            circ0 = QuantumCircuit(2, 1)
            circ1 = QuantumCircuit(2, 1)

            # Stage 1: Control qubit starting in |0> state, flipping to |1>in middle
            circ0.barrier()
            circ0.sx(
                0
            )  # Put target qubit bloch vector on the equator, parallel to -y axis

            if dt_unit:
                circ0.delay(round(delay_dt / 2), 0, "dt")
            else:
                circ0.delay(delay / 2, 0, "s")

            circ0.barrier()
            circ0.x(0)  # Put target qubit bloch vector parallel to +y axis
            circ0.x(1)  # Put control qubit in |1> state
            circ0.barrier()

            if dt_unit:
                circ0.delay(round(delay_dt / 2), 0, "dt")
            else:
                circ0.delay(delay / 2, 0, "s")

            # Rotate the bloch vector of the target qubit in angle increments around the z axis, with tip of bloch vector tracing along equator
            circ0.rz(rotation_angle, 0)

            circ0.sx(0)  # Trying to put target into |1> state

            circ0.barrier()
            circ0.x(
                1
            )  # reset the control (in perfect system, makes no difference, but this puts control qubit in measurement between Stage 1 and Stage 2 of the experiment)
            circ0.barrier()

            circ0.measure(0, 0)

            # Control Target Operations

            # Stage 2: Control starting in |1> state, flipping to |0> in middle
            circ1.barrier()
            circ1.x(1)
            circ1.barrier()
            circ1.sx(0)

            if dt_unit:
                circ1.delay(round(delay_dt / 2), 0, "dt")
            else:
                circ1.delay(delay / 2, 0, "s")

            circ1.barrier()
            circ1.x(0)
            circ1.x(1)
            circ1.barrier()

            if dt_unit:
                circ1.delay(round(delay_dt / 2), 0, "dt")
            else:
                circ1.delay(delay / 2, 0, "s")

            circ1.rz(rotation_angle, 0)
            circ1.sx(0)

            circ1.barrier()

            circ1.measure(0, 0)

            circ0.metadata = metadata.copy()
            circ1.metadata = metadata.copy()
            circ0.metadata["series"] = "0"
            circ1.metadata["series"] = "1"
            circ0.metadata["xval"] = real_delay_in_sec
            circ1.metadata["xval"] = real_delay_in_sec

            circs.extend([circ0, circ1])

        return circs
