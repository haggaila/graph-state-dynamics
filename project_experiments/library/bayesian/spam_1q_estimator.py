# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from typing import Dict, List, Optional
import numpy as np
from qiskit import QuantumCircuit
from .mc_cube import MCCube

# pylint: disable=invalid-name
# pylint: disable=(invalid-unary-operand-type
# pylint: disable=consider-using-in


class BayesianSPAMEstimator:
    """This class implements the estimation of single-qubit SPAM and some gate error parameters."""

    _I_PI_X = 0
    _I_PI_Y = 1
    _I_PI_Z = 2
    _I_PI_0 = 3

    _I_X_0 = 4
    _I_Y_0 = 5
    _I_Z_0 = 6  # Note that _I_Z_0 > _I_X_0, _I_Y_0 must be respected for the code below

    _I_EPSILON = 7
    _I_THETA = 8

    SUPPORTED_GATES = [
        "id",
        "x",
        "x90p",
        "x90m",
        "y90p",
        "y90m",
        "x90p^2",
        "x90p^5",
        "x90m^5",
        "y90p^5",
        "y90m^5",
        "x90p^4n",
        "x90p^(4n+1)",
    ]
    """Names of the calibrated gates supported for the experiments."""

    SUPPORTED_PARAMETERS = {
        "pi_x": _I_PI_X,
        "pi_y": _I_PI_Y,
        "pi_z": _I_PI_Z,
        "pi_0": _I_PI_0,
        "x_0": _I_X_0,
        "y_0": _I_Y_0,
        "z_0": _I_Z_0,
        "epsilon_x90p": _I_EPSILON,
        "theta_x90p": _I_THETA,
    }
    """Names of the parameters that are supported for estimation."""

    NUM_SUPPORTED_PARAMS = len(SUPPORTED_PARAMETERS)
    """The number of parameters that are supported for estimation (as listed in
    SUPPORTED_PARAMETERS)."""

    PARAM_BOUNDARIES = {
        "pi_x": (-1.0, 1.0),
        "pi_y": (-1.0, 1.0),
        "pi_z": (-1.0, 1.0),
        "pi_0": (0.0, 1.0),
        "x_0": (-1.0, 1.0),
        "y_0": (-1.0, 1.0),
        "z_0": (-1.0, 1.0),
        "epsilon_x90p": (0.0, 1.0),
        "theta_x90p": (-np.pi / 4, np.pi / 4),
    }
    """Boundaries of the intervals over which the prior (uniform) distribution of each parameter can
    be defined."""

    BAYESIAN_QPCM_PARAMETERS = ["x_0", "y_0", "z_0", "pi_z", "pi_0"]
    """The five parameters  for estimation of Quantum Preparation and Classical Measurement errors."""

    BAYESIAN_QPCM_PRIORS = [
        [-0.1, 0.1],
        [-0.1, 0.1],
        [0.82, 1.0],
        [0.4, 0.55],
        [0.45, 0.6],
    ]
    """Five default parameter priors for estimation corresponding to the parameters defined in
    BAYESIAN_QPCM_PARAMETERS. May not be suitable for all devices, if their manifested errors
    are too large."""

    BAYESIAN_QPCMp5_PRIORS = [
        [-0.2, 0.2],
        [-0.2, 0.2],
        [0.82, 1.0],
        [0.4, 0.55],
        [0.45, 0.6],
    ]
    """Five default parameter priors for estimation corresponding to the parameters defined in
    BAYESIAN_QPCM_PARAMETERS. May not be suitable for all devices, if their manifested errors
    are too large."""

    BAYESIAN_QPCMG_PARAMETERS = [
        "x_0",
        "y_0",
        "z_0",
        "pi_z",
        "pi_0",
        "epsilon_x90p",
        "theta_x90p",
    ]
    """The seven parameters supported for estimation of Quantum Preparation / Classical Measurement
    and Gate errors."""

    BAYESIAN_QPCMG_PRIORS = [
        [-0.1, 0.1],
        [-0.1, 0.1],
        [0.82, 1.0],
        [0.4, 0.55],
        [0.45, 0.6],
        [0.0, 0.01],
        [-0.01, 0.01],
    ]
    """Seven default parameter priors for estimation corresponding to parameters defined in
    BAYESIAN_QPCMG_PARAMETERS. May not be suitable for all devices, if their manifested errors
    are too large."""

    BAYESIAN_CPCMG_PARAMETERS = ["z_0", "pi_z", "pi_0", "epsilon_x90p", "theta_x90p"]
    """The five parameters supported for estimation of Classical Preparation / Classical Measurement
    and Gate errors."""

    BAYESIAN_CPCMG_PRIORS = [
        [0.82, 1.0],
        [0.4, 0.55],
        [0.45, 0.6],
        [0.0, 0.01],
        [-0.02, 0.02],
    ]
    """Seven default parameter priors for estimation corresponding to parameters defined in
    BAYESIAN_CPCMG_PARAMETERS. May not be suitable for all devices, if their manifested errors
    are too large."""

    BAYESIAN_DIRECT_GATES = ["id", "x", "x90p", "x90m", "y90p", "y90m"]
    """The six nonconcatenated gates used for estimation using a Bayesian estimation, without
    gate errors."""

    BAYESIAN_QPCM_GATES = ["id", "x", "x90p", "x90m", "y90p", "y90m"]
    """The six nonconcatenated gates used for estimation using a Bayesian estimation, without
    gate errors."""

    BAYESIAN_QPCMp5_GATES = ["id", "x90p^2", "x90p^5", "x90m^5", "y90p^5", "y90m^5"]
    """The six nonconcatenated gates used for estimation using a Bayesian estimation, without
    gate errors."""

    BAYESIAN_CPCMG_GATES = ["id", "x90p^2", "x90p", "x90m", "x90p^4n", "x90p^(4n+1)"]
    """The six  gates used for estimation using a Bayesian estimation, without gate errors."""

    BAYESIAN_FULL_GATES = [
        "id",
        "x90p^2",
        "x90p",
        "x90m",
        "y90p",
        "y90m",
        "x90p^4n",
        "x90p^(4n+1)",
    ]
    """The eight gates used for estimation of QPCMG/CPCMG parameters using a Bayesian estimation."""

    DEFAULT_X90P_POWER = 8

    def __init__(
        self,
        gates: List[str],
        parameters: List[str],
        prior_intervals: Optional[List[List[float]]],
        n_draws,
        n_x90p_power=0,
        n_repeats=1,
    ):
        """
        Args:
                gates: A list with the gates to be used for the estimation.
                        The supported gates are given in the member list self.SUPPORTED_GATES.
                parameters: A list with the string names of the parameters that are to be estimated.
                        The supported parameters are given in the list self.SUPPORTED_PARAMETERS.
                        The order of the parameters in the parameters list determines the order of
                        entries in the output arrays.
                prior_intervals: A list corresponding to each of the parameters in argument 'parameters',
                        with each entry being a list of two elements, the lower and upper boundaries
                        of the prior distribution to be used in Bayesian estimation of the
                        corresponding parameter.
                n_draws: The number of points to draw in the Bayesian estimation.
                n_x90p_power: The power n with which the application of the x90p gate was repeated for
                        estimating gate error parameters theta and epsilon (applicable for the supported
                        gates 'x90p^4n' and 'x90p^(4n+1). If left at 0, the member DEFAULT_X90P_POWER
                        is used.
                n_repeats: The number of times to repeat the circuits, collecting all shots together.
        """
        self.gates = gates
        self.parameters = parameters
        self.prior_intervals = prior_intervals
        self.n_draws = int(n_draws)
        if n_x90p_power == 0:
            n_x90p_power = self.DEFAULT_X90P_POWER
        self.n_x90p_power = n_x90p_power
        self.n_repeats = n_repeats
        self.cube = None

    def prepare_Bayesian(self):
        """Precomputes the Monte Carlo cube used for Bayesian estimation, enforcing constraints.
        With Python being an interpreted language and this computation being a bottle neck of the
        estimation performance (handling float arrays with Gigas of parameter draws), the code below
        is expanded/unfolded and vectorized as much as possible, to reduce the number of repeated
        computations and achieve the maximal efficiency (while using more intermediate memory).
        An MCCube class describing the Monte Carlo calculation, that can be used for the estimation,
        is constructed and stored internally.
        """
        gates = self.gates
        parameters = self.parameters
        prior_intervals = self.prior_intervals
        n_draws = self.n_draws
        n_x90p_power = self.n_x90p_power

        param_indices = [-1] * BayesianSPAMEstimator.NUM_SUPPORTED_PARAMS
        for i_param, s_param in enumerate(parameters):
            param_indices[
                BayesianSPAMEstimator.SUPPORTED_PARAMETERS[s_param.lower()]
            ] = i_param

        if len(prior_intervals) != len(parameters):
            raise Exception(
                "The argument prior_intervals must match the parameters argument."
            )

        V = 1.0
        for i_param, s_param in enumerate(parameters):
            lb, ub = BayesianSPAMEstimator.PARAM_BOUNDARIES[s_param]
            interval = prior_intervals[i_param]
            if interval[1] < interval[0]:
                raise Exception(
                    f"The prior interval for parameter {s_param} is ill-defined."
                )
                # We raise exception in this case because such an ambiguity should be fixed by the user.
            b_interval_fixed = False
            if interval[1] > ub:
                interval[1] = ub
                b_interval_fixed = True
            if interval[0] < lb:
                interval[0] = lb
                b_interval_fixed = True
            if b_interval_fixed:
                print(
                    f"Prior interval for parameter {s_param} was restricted to parameter bounds."
                )
            V *= interval[1] - interval[0]

        # print(f"Preparing Monte Carlo cube with {n_draws:,} parameter draws.")
        cube = MCCube(prior_intervals, V)
        cube.draw(n_draws)
        ordered_values = []
        ordered_indices = []
        b_x0_y0 = False
        for i_param, param_index in enumerate(param_indices):
            if param_index != -1:
                ordered_indices.append(param_index)
                ordered_values.append(cube.values[param_index])
            elif i_param != BayesianSPAMEstimator._I_Z_0:
                ordered_indices.append(-1)
                ordered_values.append(cube.zeros)
                if (
                    i_param == BayesianSPAMEstimator._I_X_0
                    or i_param == BayesianSPAMEstimator._I_Y_0
                ):
                    b_x0_y0 = True
            else:  # z_0 is the only one that defaults to 1 if not being estimated
                if (
                    b_x0_y0
                ):  # This requires BayesianSPAMEstimator._I_Z_0 > BayesianSPAMEstimator._I_X_0 and Y0 !
                    raise Exception(
                        "If either x_0 or y_0 parameters are defined for estimation, z_0 must be"
                        " estimated as well for the consistency of the Bloch vector."
                    )
                ordered_indices.append(-2)
                ordered_values.append(cube.ones)
        cube.ordered_indices = ordered_indices
        cube.ordered_values = ordered_values
        pis = (
            BayesianSPAMEstimator._I_PI_X,
            BayesianSPAMEstimator._I_PI_Y,
            BayesianSPAMEstimator._I_PI_Z,
        )
        r0s = (
            BayesianSPAMEstimator._I_X_0,
            BayesianSPAMEstimator._I_Y_0,
            BayesianSPAMEstimator._I_Z_0,
        )
        ipi0 = BayesianSPAMEstimator._I_PI_0
        ipi_z = BayesianSPAMEstimator._I_PI_Z

        # print(f"Validations.")
        # VALIDATIONS
        sum_pi_a_2 = (
            (ordered_values[pis[0]] ** 2)
            + (ordered_values[pis[1]] ** 2)
            + (ordered_values[pis[2]] ** 2)
        )
        sum_r_0_2 = (
            (ordered_values[r0s[0]] ** 2)
            + (ordered_values[r0s[1]] ** 2)
            + (ordered_values[r0s[2]] ** 2)
        )
        conditions = (
            (ordered_values[ipi0] + ordered_values[ipi_z]) > 1.0,
            (ordered_values[ipi0] + ordered_values[ipi_z]) <= 0.0,
            (ordered_values[ipi0] - ordered_values[ipi_z]) >= 1.0,
            (ordered_values[ipi0] + ordered_values[ipi_z]) < 0.0,
            sum_r_0_2 > 1.0,
            ordered_values[ipi0] ** 2 < sum_pi_a_2,
            (1.0 - ordered_values[ipi0]) ** 2 < sum_pi_a_2,
        )
        invalid_ps = np.full_like(ordered_values[ipi0], False, dtype=bool)
        for condition in conditions:
            invalid_ps = np.logical_or(invalid_ps, condition)
        n_valid = n_draws - np.count_nonzero(invalid_ps)
        if n_valid != n_draws:
            print(f"Using {n_valid:,} valid parameter draws.")
            cube.delete_values(invalid_ps)

        # print(f"Calculations.")
        ones_ = cube.ones
        vals = ordered_values
        ix0 = BayesianSPAMEstimator._I_X_0
        iy0 = BayesianSPAMEstimator._I_Y_0
        iz0 = BayesianSPAMEstimator._I_Z_0
        ipix = BayesianSPAMEstimator._I_PI_X
        ipiy = BayesianSPAMEstimator._I_PI_Y
        ipiz = BayesianSPAMEstimator._I_PI_Z
        itheta = BayesianSPAMEstimator._I_THETA
        iepsilon = BayesianSPAMEstimator._I_EPSILON
        vals_pi0 = vals[BayesianSPAMEstimator._I_PI_0]
        b_gate_errors = ("theta_x90p" in parameters) or ("epsilon_x90p" in parameters)
        ct = None
        st = None
        c2t = None
        s2t = None
        c4nt = None
        s4nt = None
        c4n1t = None
        s4n1t = None
        err = None
        err2 = None
        err4n = None
        err4n1 = None
        if b_gate_errors:
            if n_x90p_power == 0:
                n_x90p_power = BayesianSPAMEstimator.DEFAULT_X90P_POWER
            _4n = (
                4 * n_x90p_power
            )  # the current number of X90p concatenation used for gate parameters estimation.
            err = ones_ - vals[iepsilon]
            err2 = (ones_ - vals[iepsilon]) ** 2
            err4n = (ones_ - vals[iepsilon]) ** _4n
            err4n1 = (ones_ - vals[iepsilon]) ** (_4n + 1)
            ct = np.cos(vals[itheta])
            st = np.sin(vals[itheta])
            c2t = np.cos(2.0 * vals[itheta])
            s2t = np.sin(2.0 * vals[itheta])
            c4nt = np.cos(_4n * vals[itheta])
            s4nt = np.sin(_4n * vals[itheta])
            c4n1t = np.cos((_4n + 1) * vals[itheta])
            s4n1t = np.sin((_4n + 1) * vals[itheta])

        # print(f"Gate looping.")
        for i_gate, s_gate in enumerate(gates):
            p = None
            if b_gate_errors:
                if s_gate == "id":
                    p = (
                        vals_pi0
                        + vals[ipix] * vals[ix0]
                        + vals[ipiy] * vals[iy0]
                        + vals[ipiz] * vals[iz0]
                    )
                elif s_gate[0:3] == "y90":
                    pm = 1.0 if s_gate[3] == "p" else -1.0
                    p = (
                        vals_pi0
                        + err * (-st * vals[ix0] + pm * ct * vals[iz0]) * vals[ipix]
                        + err * vals[ipiy] * vals[iy0]
                        + err * (-pm * ct * vals[ix0] - st * vals[iz0]) * vals[ipiz]
                    )
                elif s_gate[0:3] == "x90":
                    if len(s_gate) == 4:
                        pm = 1.0 if s_gate[3] == "p" else -1.0
                        p = (
                            vals_pi0
                            + err * vals[ipix] * vals[ix0]
                            + err * (-st * vals[iy0] - pm * ct * vals[iz0]) * vals[ipiy]
                            + err * (pm * ct * vals[iy0] - st * vals[iz0]) * vals[ipiz]
                        )
                    elif s_gate == "x90p^2":
                        p = (
                            vals_pi0
                            + err2 * vals[ipix] * vals[ix0]
                            + err2 * (-c2t * vals[iy0] + s2t * vals[iz0]) * vals[ipiy]
                            + err2 * (-s2t * vals[iy0] - c2t * vals[iz0]) * vals[ipiz]
                        )
                    elif s_gate == "x90p^4n":
                        p = (
                            vals_pi0
                            + err4n * vals[ipix] * vals[ix0]
                            + err4n * (c4nt * vals[iy0] - s4nt * vals[iz0]) * vals[ipiy]
                            + err4n * (s4nt * vals[iy0] + c4nt * vals[iz0]) * vals[ipiz]
                        )
                    elif s_gate == "x90p^(4n+1)":
                        p = (
                            vals_pi0
                            + err4n1 * vals[ipix] * vals[ix0]
                            + err4n1
                            * (-s4n1t * vals[iy0] - c4n1t * vals[iz0])
                            * vals[ipiy]
                            + err4n1
                            * (c4n1t * vals[iy0] - s4n1t * vals[iz0])
                            * vals[ipiz]
                        )
                    else:
                        raise Exception(
                            f"Unknown/unsupported gate {s_gate} for estimation."
                        )
            else:
                if s_gate == "id" or s_gate == "x90p^4n":
                    p = (
                        vals_pi0
                        + vals[ipix] * vals[ix0]
                        + vals[ipiy] * vals[iy0]
                        + vals[ipiz] * vals[iz0]
                    )
                elif s_gate == "x90p" or s_gate == "x90p^(4n+1)" or s_gate == "x90p^5":
                    p = (
                        vals_pi0
                        + vals[ipix] * vals[ix0]
                        - vals[ipiy] * vals[iz0]
                        + vals[ipiz] * vals[iy0]
                    )
                elif s_gate == "x90m" or s_gate == "x90m^5":
                    p = (
                        vals_pi0
                        + vals[ipix] * vals[ix0]
                        + vals[ipiy] * vals[iz0]
                        - vals[ipiz] * vals[iy0]
                    )
                elif s_gate == "y90p" or s_gate == "y90p^5":
                    p = (
                        vals_pi0
                        + vals[ipix] * vals[iz0]
                        + vals[ipiy] * vals[iy0]
                        - vals[ipiz] * vals[ix0]
                    )
                elif s_gate == "y90m" or s_gate == "y90m^5":
                    p = (
                        vals_pi0
                        - vals[ipix] * vals[iz0]
                        + vals[ipiy] * vals[iy0]
                        + vals[ipiz] * vals[ix0]
                    )
                elif s_gate == "x" or s_gate == "x90p^2":
                    p = (
                        vals_pi0
                        + vals[ipix] * vals[ix0]
                        - vals[ipiy] * vals[iy0]
                        - vals[ipiz] * vals[iz0]
                    )
            if p is None:
                raise Exception(
                    f"Unknown/unsupported gate {s_gate} for estimation (note that 'x' gate is not "
                    "supported together with gate errors."
                )
            log_p = p * 0.0
            p_pos = p > 0.0
            log_p[p_pos] = np.log(p[p_pos])
            p_non_positive = np.logical_not(p_pos)
            if np.count_nonzero(p_non_positive) == 0:
                p_non_positive = None
            p_non_1 = p < 1.0
            log_1_p = p * 0.0
            log_1_p[p_non_1] = np.log(1.0 - p[p_non_1])
            p_non_ones = np.logical_not(p_non_1)
            if np.count_nonzero(p_non_ones) == 0:
                p_non_ones = None

            key = (s_gate, i_gate)
            cube.log_p[key] = log_p
            cube.log_1_p[key] = log_1_p
            cube.p_non_positive[key] = p_non_positive
            cube.p_non_ones[key] = p_non_ones
            # cube.probabilities[key] = p  # commented out to reduce memory usage
        self.cube = cube

    def estimate_Bayesian(
        self,
        gate_counts: Dict,
        b_full_covariances: bool = False,
        b_full_distributions: bool = False,
    ) -> Dict:
        """Estimate the SPAM/gate error parameters of a single qubit, and their variances.

        Args:
                gate_counts: A dict with string keys of the calibration gates that were run.
                        Each keyed entry consists of gate_counts is a two-element tuple, the first
                        is the marginalized counts of the result 0, and the second of the result 1.
                        The total number of shots for all gate counts must be identical.
                b_full_covariances: If True, the full covariance matrix of the parameters is estimated.
                        Otherwise, only the diagonal elements that correspond to the parameter
                        variances are calculated. [CURRENTLY UNIMPLEMENTED]
                b_full_distributions: If True and the estimation is Bayesian, the result dictionary
                        stores the full distribution information used in the calculation (which may
                        be very large).

        Returns:
                A dictionary with the results and intermediate calculations.
                Among other entries, the dictionary contains a vector of the means estimated for each
                parameter, and a matrix of covariances. If the estimation is Bayesian an MCCube class
                describing the Monte Carlo calculation is returned as well, and further data generated
                during the estimation.
        Raises:
            Exception: in case of inconsistent cube or full unsupported options requested.
        """

        parameters: List[str] = self.parameters
        cube: MCCube = self.cube
        n_valid_draws = len(cube.ones)
        log_L = np.zeros(n_valid_draws)

        for key in gate_counts.keys():
            cc = gate_counts.get(key)
            log_p = cube.log_p.get(key, None)
            if log_p is None:
                raise Exception("Inconsistent Bayesian preparation cube.")
            log_1_p = cube.log_1_p.get(key, None)
            p_non_positive = cube.p_non_positive.get(key, None)
            p_non_ones = cube.p_non_ones.get(key, None)
            if cc[0] > 0:
                log_L += log_p * cc[0]
                if p_non_positive is not None:
                    log_L[p_non_positive] = -np.inf
            if cc[1] > 0:
                log_L += log_1_p * cc[1]
                if p_non_ones is not None:
                    log_L[p_non_ones] = -np.inf

        log_L -= np.max(log_L)
        V_inv = 1.0 / cube.volume
        P = np.exp(log_L) * V_inv
        N = np.nansum(P / V_inv) / n_valid_draws
        P /= N
        Var_P = np.nansum(np.square(P / V_inv - 1.0)) / (
            n_valid_draws * (n_valid_draws - 1)
        )
        # print(f"The sample estimate of the variance of the integral of the posterior distribution "
        # 		   f"is\n\tVar[P] = {Var_P:5}.")

        n_params = len(parameters)
        mean = np.zeros((n_params,))
        cov = np.full((n_params, n_params), np.nan)
        # Init to nan, will be overwritten if covariances are calculated
        mean_dict = {}
        vars_dict = {}
        for i_param, s_param in enumerate(parameters):
            m = (
                np.nansum(P * cube.values[i_param] / V_inv) / n_valid_draws
            )  # mean estimator
            mm = (
                np.nansum(P * np.square(cube.values[i_param]) / V_inv) / n_valid_draws
            )  # 2nd-moment estimator
            v = mm - m**2  # variance estimator if MC integral were exact
            # vs = np.nansum(np.square(P * cube.values[i_param] / V_inv - m)) /
            #                (n_valid_draws * (n_draws - 1))
            # The above is the sample variance estimate of the integral variance
            mean_dict[s_param] = m
            vars_dict[s_param] = v
            # stddev = v**0.5
            # s_estimate = (
            #     f"Estimated {s_param}: mean {round(m, 5)}, std dev: {round(stddev, 5)}."
            # )
            # print(s_estimate)

        if b_full_covariances:
            raise Exception("Full covariances are currently not implemented.")
        # 	Skeleton, needs fixing:
        # 	for i_param1, _ in enumerate(parameters):
        # 		for i_param2 in range(i_param1):
        # 			mm = np.sum(P * np.square(cube.values[i_param1]) / P_MC) / n_draws
        # 			vv = mm
        # 			cov[i_param1, i_param2] = vv
        # 			cov[i_param2, i_param1] = vv

        result = {
            "mean": mean,
            "cov": cov,
            "Var_P": Var_P,
            "n_valid_draws": n_valid_draws,
            "mean_dict": mean_dict,
            "vars_dict": vars_dict,
        }
        if b_full_distributions:
            result["cube"] = cube
            result["P"] = P
        return result

    # def get_1q_circuits(self, qubit) -> List[QuantumCircuit]:
    #     """Return a list of experiment circuits.
    #
    #     Returns:
    #         A list of :class:`QuantumCircuit`.
    #
    #     Raises:
    #         Exception: In case of unsupported gates requested.
    #     """
    #     gates = self.gates
    #
    #     circuits = []
    #     pi_2 = np.pi / 2
    #     for _, s_gate in enumerate(gates):
    #         circ = QuantumCircuit(1, 1)
    #
    #         if s_gate == "x":
    #             circ.x(qubit)
    #         elif s_gate[0:4] == "x90p":
    #             if s_gate == "x90p":
    #                 n_len = 1
    #             elif s_gate == "x90p^2":
    #                 n_len = 2
    #             elif s_gate == "x90p^4n":
    #                 n_len = 4 * self.n_x90p_power
    #             elif s_gate == "x90p^(4n+1)":
    #                 n_len = 4 * self.n_x90p_power + 1
    #             else:
    #                 raise Exception(f"Unknown/unsupported instruction {s_gate}.")
    #             for _ in range(n_len):
    #                 circ.rx(pi_2, qubit)
    #         elif s_gate == "x90m":
    #             circ.rx(-pi_2, qubit)
    #         elif s_gate == "y90p":
    #             circ.ry(pi_2, qubit)
    #         elif s_gate == "y90m":
    #             circ.ry(pi_2, -qubit)
    #         elif s_gate == "id":
    #             pass
    #         else:
    #             raise Exception(f"Unknown/unsupported instruction {s_gate}.")
    #         circ.measure(0, 0)
    #
    #         circ.metadata = {"experiment_type": "Bayesian-spam-1q", "qubits": [qubit]}
    #         circuits.append(circ)
    #
    #     return circuits
