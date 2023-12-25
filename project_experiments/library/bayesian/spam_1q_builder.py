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

from typing import Optional, Sequence

from qiskit.providers import Backend

from qiskit_experiments.framework import (
    BaseExperiment,
    ParallelExperiment,
)

from project_experiments.partition import partition_qubits

from .spam_1q_estimator import (
    BayesianSPAMEstimator,
)
from .spam_1q_experiment import (
    BayesianSPAMExperiment,
)


class BayesianSPAMBuilder:
    """
    Build a set of parallel experiments, consisting of Bayesian estimation experiments.

    Args:
        gates: A list with the gates to be used for the estimation.
            The supported gates are given in the member list self.SUPPORTED_GATES.
        parameters: A list with the string names of the parameters that are to be estimated.
            The supported parameters are given in the list self.SUPPORTED_PARAMETERS.
            The order of the parameters in the parameters list determines the order of entries
            in the output arrays.
        prior_intervals: A list corresponding to each of the parameters in argument 'parameters',
            with each entry being a list of two elements, the lower and upper boundaries of the
            prior distribution to be used in Bayesian estimation of the corresponding parameter.
        n_draws: The number of points to draw in the Bayesian estimation.
        n_x90p_power: The power n with which the application of the x90p gate was repeated for
            estimating gate error parameters theta and epsilon (applicable for the supported
            gates 'x90p^4n' and 'x90p^(4n+1). If left at 0, the member DEFAULT_X90P_POWER is used.
        experiment_options: Experiment options.
        analysis_options: Analysis options.
        transpile_options: Transpile options.
        distance: The graph distance parameter for parallelizing the qubits. A value of 1 indicates
            all qubits in parallel, a value of 2 indicates next-nearest-neighbors are parallel, etc.
        s_add_suffix: An optional suffix string to add to the experiment results.
        b_pulse_gates: Whether to use pulse gates. Important if gate errors are not small
            enough, and in order to meaningfully estimate gate errors.
    """

    def __init__(
        self,
        gates=None,
        parameters=None,
        prior_intervals=None,
        n_draws=int(20e6),
        n_x90p_power=8,
        n_repeats=1,
        experiment_options: Optional[dict] = None,
        analysis_options: Optional[dict] = None,
        transpile_options: Optional[dict] = None,
        user_qubit_groups: Optional[Sequence[Sequence[Sequence[int]]]] = None,
        distance=2,
        s_add_suffix: Optional[str] = "",
        b_pulse_gates=False,
    ):
        if gates is None:
            gates = BayesianSPAMEstimator.BAYESIAN_CPCMG_GATES
        if parameters is None:
            parameters = BayesianSPAMEstimator.BAYESIAN_CPCMG_PARAMETERS
        if prior_intervals is None:
            prior_intervals = BayesianSPAMEstimator.BAYESIAN_CPCMG_PRIORS
        self.gates = gates
        self.parameters = parameters
        self.prior_intervals = prior_intervals
        self.n_draws = n_draws
        self.n_x90p_power = n_x90p_power
        self.n_repeats = n_repeats
        self._experiment_options = experiment_options
        self._analysis_options = analysis_options
        if transpile_options is None:
            transpile_options = {"optimization_level": 0}
        self._transpile_options = transpile_options
        self._user_qubit_groups = user_qubit_groups
        self._distance = distance
        self._s_add_suffix = s_add_suffix
        self._b_pulse_gates = b_pulse_gates

    def build(self, backend: Backend, model=None) -> [BaseExperiment]:
        """Build the batch of parallel experiments according to the qubit groups, constructed
            according to the requested `distance` parameter of the constructor.

        Args:
            backend: The backend whose connectivity is used to parallelize the experiments.
            model: An optional BayesianSPAMEstimator to use - if None, one will be created
                and initialized.

        Returns:
            The experiment for the device.
        """
        if model is None:
            model = BayesianSPAMEstimator(
                gates=self.gates,
                parameters=self.parameters,
                prior_intervals=self.prior_intervals,
                n_draws=self.n_draws,
                n_x90p_power=self.n_x90p_power,
                n_repeats=self.n_repeats,
            )
            model.prepare_Bayesian()

        qubit_groups = self._user_qubit_groups or partition_qubits(
            backend, self._distance
        )
        faulty_qubits = backend.properties().faulty_qubits()
        par_exps = []
        for group in qubit_groups:
            exps = []
            for qubit in group:
                if qubit[0] in faulty_qubits:
                    continue
                exp = BayesianSPAMExperiment(
                    qubit=qubit[0],
                    model=model,
                    add_suffix=self._s_add_suffix,
                    b_pulse_gates=self._b_pulse_gates,
                )
                if self._experiment_options:
                    exp.set_experiment_options(**self._experiment_options)
                if self._analysis_options:
                    exp.analysis.set_options(**self._analysis_options)
                exps.append(exp)
            par_exp = ParallelExperiment(exps)
            if self._transpile_options:
                par_exp.set_transpile_options(**self._transpile_options)
            par_exps.append(par_exp)
        return par_exps
