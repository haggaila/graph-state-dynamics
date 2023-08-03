# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Ramsey XY frequency characterization experiment."""

from typing import List, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.providers.fake_provider import FakeBackend

from project_experiments.library.ramsey_xy_analysis import RamseyXYAnalysis
from qiskit_experiments.framework import BaseExperiment
from qiskit_experiments.framework.restless_mixin import RestlessMixin


class RamseyXY(BaseExperiment, RestlessMixin):
    r"""Ramsey XY experiment to measure the frequency of a qubit.

    # section: overview

        This experiment differs from the :class:`~qiskit_experiments.characterization.\
        t2ramsey.T2Ramsey` since it is sensitive to the sign of the frequency offset from
        the main transition. This experiment consists of following two circuits:

        .. parsed-literal::

            (Ramsey X) The second pulse rotates by pi-half around the X axis
                        OUTDATED
                       ┌────┐┌─────────────┐┌───────┐┌────┐ ░ ┌─┐
                  q_0: ┤ √X ├┤ Delay(τ[s]) ├┤ Rz(θ) ├┤ √X ├─░─┤M├
                       └────┘└─────────────┘└───────┘└────┘ ░ └╥┘
            measure: 1/════════════════════════════════════════╩═
                                                               0

            (Ramsey Y) The second pulse rotates by pi-half around the Y axis
                        OUTDATED
                       ┌────┐┌─────────────┐┌───────────┐┌────┐ ░ ┌─┐
                  q_0: ┤ √X ├┤ Delay(τ[s]) ├┤ Rz(θ-π/2) ├┤ √X ├─░─┤M├
                       └────┘└─────────────┘└───────────┘└────┘ ░ └╥┘
            measure: 1/════════════════════════════════════════════╩═
                                                                   0

        The first and second circuits measure the expectation value along the X and Y axis,
        respectively. This experiment therefore draws the dynamics of the Bloch vector as a
        Lissajous figure. Since the control electronics tracks the frame of qubit at the
        reference frequency, which differs from the true qubit frequency by :math:`\Delta\omega`,
        we can describe the dynamics of two circuits as follows. The Hamiltonian during the
        ``Delay`` instruction is :math:`H^R = - \frac{1}{2} \Delta\omega` in the rotating frame,
        and the propagator will be :math:`U(\tau) = \exp(-iH^R\tau)` where :math:`\tau` is the
        duration of the delay. By scanning this duration, we can get

        .. math::

            {\cal E}_x(\tau)
                = {\rm Re} {\rm Tr}\left( Y U \rho U^\dagger \right)
                &= - \cos(\Delta\omega\tau) = \sin(\Delta\omega\tau - \frac{\pi}{2}), \\
            {\cal E}_y(\tau)
                = {\rm Re} {\rm Tr}\left( X U \rho U^\dagger \right)
                &= \sin(\Delta\omega\tau),

        where :math:`\rho` is prepared by the first :math:`\sqrt{\rm X}` gate. Note that phase
        difference of these two outcomes :math:`{\cal E}_x, {\cal E}_y` depends on the sign and
        the magnitude of the frequency offset :math:`\Delta\omega`. By contrast, the measured
        data in the standard Ramsey experiment does not depend on the sign of :math:`\Delta\omega`,
        i.e. :math:`\cos(-\Delta\omega\tau) = \cos(\Delta\omega\tau)`.

        The experiment also allows users to add a small frequency offset to better resolve
        any oscillations. This is implemented by a virtual Z rotation in the circuits. In the
        circuit above it appears as the delay-dependent angle θ(τ).

    # section: analysis_ref
        :py:class:`RamseyXYAnalysis`
    """

    @classmethod
    def _default_experiment_options(cls):
        """Default values for the Ramsey XY experiment.

        Experiment Options:
            delays (list): The list of delays that will be scanned in the experiment, in seconds.
            osc_freq (float): A frequency shift in Hz that will be applied by means of
                a virtual Z rotation to increase the frequency of the measured oscillation.
        """
        options = super()._default_experiment_options()
        options.delays = np.linspace(0, 1.0e-6, 51)
        options.osc_freq = 2e6

        return options

    def __init__(
        self,
        qubit: int,
        backend: Optional[Backend] = None,
        delays: Optional[List] = None,
        osc_freq: float = 2e6,
        b_add_z_measurement=False,
    ):
        """Create new experiment.

        Args:
            qubit: The qubit on which to run the Ramsey XY experiment.
            backend: Optional, the backend to run the experiment on.
            delays: The delays to scan, in seconds.
            osc_freq: the oscillation frequency induced by the user through a virtual
                Rz rotation. This quantity is given in Hz.
            b_add_z_measurement: Add measurement along the z axis.
        """
        super().__init__(
            [qubit], analysis=RamseyXYAnalysis(osc_freq=osc_freq), backend=backend
        )

        if delays is None:
            delays = self.experiment_options.delays
        self.set_experiment_options(delays=delays, osc_freq=osc_freq)
        self.b_add_z_measurement = b_add_z_measurement

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)

        # Scheduling parameters
        if not self._backend.configuration().simulator and not isinstance(
            backend, FakeBackend
        ):
            timing_constraints = getattr(
                self.transpile_options, "timing_constraints", {}
            )
            if "acquire_alignment" not in timing_constraints:
                timing_constraints["acquire_alignment"] = 16
            scheduling_method = getattr(
                self.transpile_options, "scheduling_method", "alap"
            )
            self.set_transpile_options(
                timing_constraints=timing_constraints,
                scheduling_method=scheduling_method,
            )

    def circuits(self) -> List[QuantumCircuit]:
        """Create the circuits for the ZZ Ramsey characterization experiment.
        Returns:
            A list of circuits with a variable delay.
        """

        metadata = {
            "experiment_type": self._type,
            "qubits": self.physical_qubits,
            "osc_freq": self.experiment_options.osc_freq,
            "unit": "s",
        }
        config = self.backend.configuration()
        if hasattr(config, "dt"):
            dt_unit = True
            dt_factor = config.dt
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

            circ = QuantumCircuit(1, 1)
            circ.barrier()
            circ.sx(0)
            circ.rz(np.pi / 2, 0)
            circ.barrier()
            if dt_unit:
                circ.delay(round(delay_dt), 0, "dt")
            else:
                circ.delay(delay, 0, "s")

            circ.barrier()
            circ.rz(rotation_angle, 0)
            circ.barrier()

            circ_x = circ.copy()
            circ_y = circ.copy()

            circ_x.rz(-np.pi / 2, 0)
            circ_x.sxdg(0)
            circ_x.barrier()
            circ_x.measure(0, 0)

            circ_x.metadata = metadata.copy()
            circ_x.metadata["series"] = "X"
            circ_x.metadata["xval"] = real_delay_in_sec

            circ_y.rz(np.pi, 0)
            circ_y.sxdg(0)
            circ_y.barrier()
            circ_y.measure(0, 0)

            circ_y.metadata = metadata.copy()
            circ_y.metadata["series"] = "Y"
            circ_y.metadata["xval"] = real_delay_in_sec

            if self.b_add_z_measurement:
                circ_z = circ.copy()
                circ_z.measure(0, 0)

                circ_z.metadata = metadata.copy()
                circ_z.metadata["series"] = "Z"
                circ_z.metadata["xval"] = real_delay_in_sec

                circs.extend([circ_x, circ_y, circ_z])
            else:
                circs.extend([circ_x, circ_y])

        return circs

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata
