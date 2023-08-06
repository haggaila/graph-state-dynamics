# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import List
import numpy as np
from qiskit_experiments.curve_analysis import CurveData


def myfft(signal, dt):
    """
    if 2D array (a,b): fft each row b.

    :param signal:
    :param dt:
    :return:
    """

    n = signal.shape[signal.ndim - 1]
    fourier_signal = dt * np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal)))
    freq_vec = np.fft.fftfreq(n, d=dt)
    freq_vec = np.fft.fftshift(freq_vec)

    return fourier_signal, freq_vec


def myifft(fourier_signal, df):
    """if 2D array (a,b): fft each row b"""

    n = fourier_signal.shape[fourier_signal.ndim - 1]
    signal = (df * n) * np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(fourier_signal)))
    time = np.fft.fftfreq(n, d=df)
    time = np.fft.fftshift(time)

    return signal, time


def n_dominant_freq(fourier_signal, freq_vec, n: int):
    """
    return n dominant frequencies in ascending order
    """
    positive_fourier_signal = fourier_signal[freq_vec > 0]
    positive_freqs = freq_vec[freq_vec > 0]
    top_n_freqs = np.sort(
        positive_freqs[np.argpartition(abs(positive_fourier_signal), -n)[-n:]]
    )
    return top_n_freqs


def combine_curve_data(
    curve_datas: List[CurveData], labels: List[str], b_same_data_allocation=False
):
    """Concatenate the data for single Curve data"""
    c_0 = curve_datas[0]
    xdata = c_0.x
    ydata = c_0.y
    sigma = c_0.y_err
    shots = c_0.shots
    data_allocation = c_0.data_allocation
    data_allocation_counter = len(c_0.labels)

    for curve_data in curve_datas[1:]:
        xdata = np.append(xdata, curve_data.x)
        ydata = np.append(ydata, curve_data.y)
        sigma = np.append(sigma, curve_data.y_err)
        shots = np.append(shots, curve_data.shots)

        d = curve_data.data_allocation

        if (
            int(np.mean(d)) == 0
        ):  # If the data allocation is 0, set proper counting index
            data_allocation = np.append(data_allocation, d + data_allocation_counter)
        else:
            data_allocation = np.append(
                data_allocation, d + data_allocation_counter - int(np.mean(d))
            )

        data_allocation_counter = data_allocation_counter + len(curve_data.labels)
    if b_same_data_allocation:
        data_allocation = np.array([0 for _ in data_allocation])
    return CurveData(
        x=xdata,
        y=ydata,
        y_err=sigma,
        shots=shots,
        data_allocation=data_allocation,
        labels=labels,
    )
