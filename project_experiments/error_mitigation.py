# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Optional

import numpy as np
import pickle
from uncertainties import unumpy as unp, ufloat
from qiskit_experiments.curve_analysis import CurveData


# unumpy shortcuts
unp_n = unp.nominal_values
unp_s = unp.std_devs


def calc_mitigation_matrix(
    p_0_given_0: Optional[ufloat],
    p_0_given_1: Optional[ufloat],
):
    """
    Calculate the readout error mitigation matrix given the probabilities p_0_given_0 and p_0_given_1
    """
    p_1_given_0 = 1 - p_0_given_0
    p_1_given_1 = 1 - p_0_given_1
    M_n = np.asarray([[p_0_given_0.n, p_0_given_1.n], [p_1_given_0.n, p_1_given_1.n]])
    M_s = np.asarray([[p_0_given_0.s, p_0_given_1.s], [p_1_given_0.s, p_1_given_1.s]])

    M = unp.uarray(M_n, M_s)

    # Compute inverse mitigation matrix
    try:
        M_inv = unp.ulinalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = unp.ulinalg.pinv(M)

    return M_inv


def mitigate_1Q_readout(
    P0_X: CurveData,
    P0_Y: CurveData,
    P0_Z: CurveData,
    p_0_given_0: Optional[ufloat] = None,
    p_0_given_1: Optional[ufloat] = None,
):
    """
    Calculate probabilities after readout error mitigation
    :return: P0_X_mit, P0_Y_mit, P0_Z_mit, M_inv
    """
    if p_0_given_0 is None:
        p_0_given_0 = ufloat(P0_X.y[0], P0_X.y_err[0])
    if p_0_given_1 is None:
        p_0_given_1 = 2 * ufloat(P0_Y.y[0], P0_Y.y_err[0]) - p_0_given_0

    M_inv = calc_mitigation_matrix(p_0_given_0, p_0_given_1)

    p0_X = unp.uarray(P0_X.y, P0_X.y_err)
    p0_Y = unp.uarray(P0_Y.y, P0_Y.y_err)
    P0_X_mit = M_inv[0, 0] * p0_X + M_inv[0, 1] * (1 - p0_X)
    P0_Y_mit = M_inv[0, 0] * p0_Y + M_inv[0, 1] * (1 - p0_Y)
    P0_Z_mit = None
    if P0_Z is not None:
        p0_Z = unp.uarray(P0_Z.y, P0_Z.y_err)
        P0_Z_mit = M_inv[0, 0] * p0_Z + M_inv[0, 1] * (1 - p0_Z)

    return P0_X_mit, P0_Y_mit, P0_Z_mit, M_inv


def mitigate_2Q_readout(prob, M_inv_q1, M_inv_q2):
    """
    return the mitigated probabilities of 2 qubits and the mitigation matrix
    """
    M_inv_2Q = np.kron(M_inv_q1, M_inv_q2)
    prob_mit = M_inv_2Q @ prob

    return prob_mit, M_inv_2Q


def mitigate_3Q_readout(prob, M_inv_q1, M_inv_q2, M_inv_q3):
    """
    return the mitigated probabilities of 3 qubits and the mitigation matrix
    """
    M_inv_3Q = np.kron(M_inv_q1, np.kron(M_inv_q2, M_inv_q3))
    prob_mit = M_inv_3Q @ prob

    return prob_mit, M_inv_3Q


def prob2ev_1Q(p_0):
    """
    Calculate expectation value of one qubit operator
    """
    ev = 2 * p_0 - 1
    return ev


def prob2ev_2Q(prob):
    """
    Calculate expectation value of two qubits operator
    """
    p_00 = prob[0, :]
    p_01 = prob[1, :]
    p_10 = prob[2, :]
    p_11 = prob[3, :]

    ev = p_00 + p_11 - p_01 - p_10

    return ev


def prob2ev_3Q(prob):
    """
    Calculate expectation value of three qubits operator
    """
    p_000 = prob[0, :]
    p_001 = prob[1, :]
    p_010 = prob[2, :]
    p_011 = prob[3, :]
    p_100 = prob[4, :]
    p_101 = prob[5, :]
    p_110 = prob[6, :]
    p_111 = prob[7, :]

    ev = p_000 - p_111 + (p_011 + p_101 + p_110) - (p_001 + p_010 + p_100)

    return ev


def calc_correlation(second_moment, first_moment_q1, first_moment_q2):
    """
    Calculate correlation
    """

    correlation = second_moment - first_moment_q1 * first_moment_q2

    return correlation


def load_correlations(formatted_data, qubits, ev_1Q_dict, M_inv_dict):
    """

    :param formatted_data: data loaded from pickle
    :param qubits: qubits to load
    :param ev_1Q_dict: dictionary of 1Q expectation values.  example: ev_1Q_dict["Q12"]
    :param M_inv_dict: dictionary of the 1Q error mitigation matrices.  example: M_inv_dict["Q5"]
    :return: dictionary the 2Q correlations.  example: correlations_dict["X_12_X_5"]
    """

    n_qubits = len(qubits)
    correlations_dict = dict()
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            for base in ("X", "Y"):
                outcome = "00"
                subset = f"{base}_{qubits[i]}_{base}_{qubits[j]}_P{outcome}"
                fd = formatted_data.get_subset_of(subset)
                p_00 = unp.uarray(fd.y, fd.y_err)

                outcome = "01"
                subset = f"{base}_{qubits[i]}_{base}_{qubits[j]}_P{outcome}"
                fd = formatted_data.get_subset_of(subset)
                p_01 = unp.uarray(fd.y, fd.y_err)

                outcome = "10"
                subset = f"{base}_{qubits[i]}_{base}_{qubits[j]}_P{outcome}"
                fd = formatted_data.get_subset_of(subset)
                p_10 = unp.uarray(fd.y, fd.y_err)

                outcome = "11"
                subset = f"{base}_{qubits[i]}_{base}_{qubits[j]}_P{outcome}"
                fd = formatted_data.get_subset_of(subset)
                p_11 = unp.uarray(fd.y, fd.y_err)

                prob_matrix_n = np.asarray(
                    [unp_n(p_00), unp_n(p_01), unp_n(p_10), unp_n(p_11)]
                )
                prob_matrix_s = np.asarray(
                    [unp_s(p_00), unp_s(p_01), unp_s(p_10), unp_s(p_11)]
                )

                prob_matrix = unp.uarray(prob_matrix_n, prob_matrix_s)

                prob_matrix_mit, _ = mitigate_2Q_readout(
                    prob_matrix,
                    M_inv_dict[f"Q{qubits[i]}"],
                    M_inv_dict[f"Q{qubits[j]}"],
                )

                second_moment = prob2ev_2Q(prob_matrix_mit)
                first_moment_q1 = ev_1Q_dict[f"{base}_{qubits[i]}"]
                first_moment_q2 = ev_1Q_dict[f"{base}_{qubits[j]}"]
                s_moment = f"{base}_{qubits[i]}_{base}_{qubits[j]}"
                s_correlation = s_moment + "_c"
                correlations_dict[s_moment] = second_moment
                correlations_dict[s_correlation] = (
                    second_moment - first_moment_q1 * first_moment_q2
                )
    return correlations_dict


def load_stabilizer(
    topo_index, xzz_list, formatted_data, qubits, M_inv_dict, b_mitigate=True
):
    """

    :param topo_index: list of indices of qubits in the chain/ring topology
    :param xzz_list: list of list of indices of the qubits in each stabilizer
    :param formatted_data: data loaded from pickle
    :param qubits: qubits to load
    :param M_inv_dict: dictionary of the 1Q error mitigation matrices.  example: M_inv_dict["Q5"]
    :return: dictionary the stabilizer expectation value.  example: correlations_dict["X_12_X_5"]
    """
    n_time_steps = int(len(formatted_data.data_allocation) / len(formatted_data.labels))
    formatted_data.data_allocation[:n_time_steps] = np.zeros((n_time_steps,))
    n_qubits = len(qubits)
    stabilizer_dict = dict()
    if n_qubits == 12:
        for i in range(n_qubits):
            prob = []
            for outcome in ("000", "001", "010", "011", "100", "101", "110", "111"):
                subset = f"stabilizer_{topo_index[i]}_P{outcome}"
                fd = formatted_data.get_subset_of(subset)
                p = unp.uarray(fd.y, fd.y_err)
                prob.append(p)

            prob_matrix_n = np.asarray([unp_n(p) for p in prob])
            prob_matrix_s = np.asarray([unp_s(p) for p in prob])

            prob_matrix = unp.uarray(prob_matrix_n, prob_matrix_s)

            if b_mitigate:
                xzz_tuple = xzz_list[i]
                prob_matrix_mit, _ = mitigate_3Q_readout(
                    prob_matrix,
                    M_inv_dict[f"Q{qubits[xzz_tuple[0]]}"],
                    M_inv_dict[f"Q{qubits[xzz_tuple[1]]}"],
                    M_inv_dict[f"Q{qubits[xzz_tuple[2]]}"],
                )
            else:
                prob_matrix_mit, _ = mitigate_3Q_readout(
                    prob_matrix, np.eye(2), np.eye(2), np.eye(2)
                )

            stabilizer_expectation = prob2ev_3Q(prob_matrix_mit)
            stabilizer_dict[f"stabilizer_{topo_index[i]}"] = stabilizer_expectation
    # 3q chain, only the middle stabilizer is relevant
    if n_qubits == 3:
        i = 1
        prob = []
        for outcome in ("000", "001", "010", "011", "100", "101", "110", "111"):
            subset = f"stabilizer_{topo_index[i]}_P{outcome}"
            fd = formatted_data.get_subset_of(subset)
            p = unp.uarray(fd.y, fd.y_err)
            prob.append(p)

        prob_matrix_n = np.asarray([unp_n(p) for p in prob])
        prob_matrix_s = np.asarray([unp_s(p) for p in prob])

        prob_matrix = unp.uarray(prob_matrix_n, prob_matrix_s)

        if b_mitigate:
            xzz_tuple = xzz_list[0]
            prob_matrix_mit, _ = mitigate_3Q_readout(
                prob_matrix,
                M_inv_dict[f"Q{qubits[xzz_tuple[0]]}"],
                M_inv_dict[f"Q{qubits[xzz_tuple[1]]}"],
                M_inv_dict[f"Q{qubits[xzz_tuple[2]]}"],
            )
        else:
            prob_matrix_mit, _ = mitigate_3Q_readout(
                prob_matrix, np.eye(2), np.eye(2), np.eye(2)
            )

        stabilizer_expectation = prob2ev_3Q(prob_matrix_mit)
        stabilizer_dict[f"stabilizer_{topo_index[i]}"] = stabilizer_expectation
    return stabilizer_dict


def load_observables(
    s_pickle_path,
    topo_index=None,
    xzz_list=None,
    b_mitigate_readout=True,
    b_correlations=False,
    p_0_given_0=None,
    p_0_given_1=None,
    b_stabilizers=False,
):
    with open(s_pickle_path + ".pkl", "rb") as handle:
        data_1Q = pickle.load(handle)

    # Import the qubits labels in the data
    qubits = [label[2:] for label in data_1Q.labels if label[:2] == "X_"]
    Minv_dict = {}
    ev_1Q_dict = {}
    times = None

    if not b_stabilizers:
        for i_qubit, qubit in enumerate(qubits):
            RamX, RamY = data_1Q.get_subset_of(f"X_{qubit}"), data_1Q.get_subset_of(
                f"Y_{qubit}"
            )
            RamZ = None
            try:
                RamZ = data_1Q.get_subset_of(f"Z_{qubit}")
            except Exception as ex:
                pass
            if i_qubit == 0:
                times = RamX.x
            Z_data = None
            if b_mitigate_readout:
                p_0_g_0, p_0_g_1 = None, None
                if p_0_given_0 is not None:
                    p_0_g_0 = p_0_given_0[i_qubit]
                    p_0_g_1 = p_0_given_1[i_qubit]
                P0_X_mit, P0_Y_mit, P0_Z_mit, Minv = mitigate_1Q_readout(
                    RamX, RamY, RamZ, p_0_g_0, p_0_g_1
                )
                Minv_dict[f"Q{qubit}"] = Minv
                X_data = prob2ev_1Q(P0_X_mit)
                Y_data = -prob2ev_1Q(P0_Y_mit)  # note minus sign to expectation value
                if P0_Z_mit is not None:
                    Z_data = prob2ev_1Q(P0_Z_mit)
            else:
                p0_X = unp.uarray(unp_n(RamX.y), unp_s(RamX.y_err))
                p0_Y = unp.uarray(unp_n(RamY.y), unp_s(RamY.y_err))
                X_data = prob2ev_1Q(p0_X)
                Y_data = -prob2ev_1Q(p0_Y)
                if RamZ is not None:
                    p0_Z = unp.uarray(unp_n(RamZ.y), unp_s(RamZ.y_err))
                    Z_data = prob2ev_1Q(p0_Z)

            ev_1Q_dict[f"X_{qubit}"] = X_data
            ev_1Q_dict[f"Y_{qubit}"] = Y_data
            ev_1Q_dict[f"Z_{qubit}"] = Z_data
    else:
        times = data_1Q.get_subset_of(data_1Q.labels[-1]).x
        if b_mitigate_readout:
            for i_qubit, qubit in enumerate(qubits):
                Minv = calc_mitigation_matrix(
                    p_0_given_0[i_qubit], p_0_given_1[i_qubit]
                )
                Minv_dict[f"Q{qubit}"] = Minv

    correlations_dict = None
    if b_correlations:
        with open(s_pickle_path + ".2Q.pkl", "rb") as handle:
            data_2Q = pickle.load(handle)
        if b_mitigate_readout:
            correlations_dict = load_correlations(
                data_2Q, qubits, ev_1Q_dict, Minv_dict
            )
        else:
            raise Exception(
                "Correlations are currently only supported with mitigation."
            )

    stabilizers_dict = None
    if b_stabilizers:
        with open(s_pickle_path + ".3Q.pkl", "rb") as handle:
            data_3Q = pickle.load(handle)
        stabilizers_dict = load_stabilizer(
            topo_index,
            xzz_list,
            data_3Q,
            qubits,
            Minv_dict,
            b_mitigate=b_mitigate_readout,
        )

    return times, qubits, ev_1Q_dict, correlations_dict, stabilizers_dict
