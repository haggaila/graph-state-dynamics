# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import List, Union, Optional, Tuple

import lmfit
import numpy as np
from qiskit.providers.options import Options

from project_experiments.library.experiment_utils import (
    myfft,
    n_dominant_freq,
    combine_curve_data,
)

from qiskit_experiments.curve_analysis.utils import eval_with_uncertainties
from qiskit_experiments.framework import AnalysisResultData
from qiskit_experiments.database_service.device_component import Qubit
from qiskit_experiments.data_processing import (
    DataProcessor,
    Probability,
    MarginalizeCounts,
)

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis import CurveData
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from uncertainties import unumpy as unp


class ZZParityAnalysis(curve.CurveAnalysis):
    r"""The ZZ Ramsey analysis is based on a fit to a cosine function.
    # section: fit_model
        Analyse a ZZ Ramsey experiment by fitting the '0' and '1' series to cosine
        functions. The two functions share the frequency and amplitude parameters
        (i.e. beta).
        .. math::
            y_0 = {\rm amp} e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq}\cdot x {\rm phase}) + {\rm base} \\
            y_1 = {\rm amp} e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq + zz}\cdot x {\rm phase}\right) + {\rm base}"
    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of both series.
            init_guess: The maximum y value less the minimum y value. 0.5 is also tried.
            bounds: [-2, 2] scaled to the maximum signal value.
        defpar \tau:
            desc: The exponential decay of the curve.
            init_guess: The initial guess is obtained by fitting an exponential to the
                square root of ('0' data)**2 + ('1' data)**2.
            bounds: [0, inf].
        defpar \rm base:
            desc: Base line of both series.
            init_guess: The average of the data. 0.5 is also tried.
            bounds: [-1, 1] scaled to the maximum signal value.
        defpar \rm freq:
            desc: Frequency of both series. This is the parameter of interest.
            init_guess: The frequency with the highest power spectral density.
            bounds: [0, inf].
        defpar \rm phase:
            desc: Common phase offset.
            init_guess: Linearly spaced between the maximum and minimum scanned beta.
            bounds: [-min scan range, max scan range].
    """

    def __init__(self, osc_freq: float = 0, cut_off_delay: Optional[int] = None):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="(amp * exp(-x / t2_0) * ( re * cos(2 * pi * (freq_e - zz) * x) + (1-re) * cos(2 * pi * (freq_o - zz) * x) ) + base)",
                    name="X|0",
                    data_sort_key={"series": "0"},
                ),
                lmfit.models.ExpressionModel(
                    expr="(amp * exp(-x / t2_1) * ( re * cos(2 * pi * (freq_e + zz) * x) + (1-re) * cos(2 * pi * (freq_o + zz) * x) ) + base)",
                    name="X|1",
                    data_sort_key={"series": "1"},
                ),
            ]
        )
        self._osc_freq = osc_freq
        self._cut_off_delay = cut_off_delay
        self._options.data_processor = DataProcessor(
            input_key="counts", data_actions=[Probability(outcome="0")]
        )

    def _run_analysis(self, experiment_data):
        analysis_results, figs = super()._run_analysis(experiment_data)

        # freq_e = next(
        #     filter(lambda res: res.name == "freq_e", analysis_results)).value
        # freq_o = next(
        #     filter(lambda res: res.name == "freq_o", analysis_results)).value
        # delta = (freq_e + freq_o) / 2 - self._osc_freq
        # nu = abs((freq_e - freq_o)) / 2
        #
        # parity_result = [
        #     AnalysisResultData(
        #         name="Delta_ZZ",
        #         value=delta,
        #         chisq=None,
        #         quality=None,
        #         extra={"unit": "Hz"}),
        #     AnalysisResultData(
        #         name="nu_ZZ",
        #         value=nu,
        #         chisq=None,
        #         quality=None,
        #         extra={"unit": "Hz"})
        # ]
        # # Return combined results
        # analysis_results = parity_result + analysis_results
        # figs[0].figure.axes[0].set_title(f"Delta={delta}, nu={nu}")
        return analysis_results, figs

    # def _run_analysis(self, experiment_data):
    #
    #     analysis_results, figs = super()._run_analysis(experiment_data)
    #
    #     def fit_function(x_values, y_values, function, init_params, bounds):
    #         fitparams, conv = curve_fit(function, x_values, y_values, init_params, bounds=bounds)
    #         y_fit = function(x_values, *fitparams)
    #
    #         return fitparams, conv, y_fit
    #
    #     def two_cos(x, amp, t2_p, freq_h, freq_l, re, base):
    #         y = amp * np.exp(-x / t2_p) * (
    #                 re * np.cos(2 * np.pi * freq_h * x) + (1 - re) * np.cos(2 * np.pi * freq_l * x)) + base
    #         return y
    #
    #     def comboFunc(comboData, amp, t2_0, t2_1, freq_h, freq_l, re, base, zz):
    #         # single data set passed in, extract separate data
    #         l = int(len(comboData) / 2)
    #         extract0 = comboData[:l]  # first data
    #         extract1 = comboData[l:]  # second data
    #
    #         result0 = two_cos(extract0, amp, t2_0, freq_h - zz, freq_l - zz, re, base)
    #         result1 = two_cos(extract1, amp, t2_1, freq_h + zz, freq_l + zz, re, base)
    #
    #         return np.append(result0, result1)
    #
    #     data_0 = self._data("0")
    #     data_1 = self._data("1")
    #
    #     data_tx, data_ty = data_0.x, data_1.x
    #     data_x, data_y = data_0.y, data_1.y
    #     data_x_err, data_y_err = data_0.y_err, data_1.y_err
    #
    #     ComboY = np.append(data_x, data_y)
    #     ComboX = np.append(data_tx, data_ty)
    #
    #     dt = data_tx[1] - data_tx[0]
    #     fourier_signal_x, freq_vec_x = myfft(data_x - np.mean(data_x), dt)
    #     fourier_signal_y, freq_vec_y = myfft(data_y - np.mean(data_y), dt)
    #
    #     positive_fourier_signal_0 = fourier_signal_x[freq_vec_x > 0]
    #     positive_fourier_signal_1 = fourier_signal_y[freq_vec_x > 0]
    #     positive_freqs = freq_vec_x[freq_vec_x > 0]
    #
    #     top_two_freqs_0 = np.sort(positive_freqs[np.argpartition(abs(positive_fourier_signal_0), -2)[-2:]])
    #     top_two_freqs_1 = np.sort(positive_freqs[np.argpartition(abs(positive_fourier_signal_1), -2)[-2:]])
    #
    #     freq_l_guess, freq_h_guess = (top_two_freqs_0 + top_two_freqs_1)/2
    #     zz_guess = np.mean((top_two_freqs_1 - top_two_freqs_0)/2)
    #
    #     fit_params, conv, y_fit = fit_function(ComboX, ComboY, comboFunc,
    #                                            init_params=[0.5, 100e-6, 100e-6, freq_h_guess,freq_l_guess, 0.5, 0.5, zz_guess],
    #                                            bounds=((0.35, 7e-6, 7e-6, -np.inf, -np.inf, 0.0, 0.4, -np.inf),
    #                                                    (0.55, 150e-6, 150e-6, np.inf, np.inf, 1, 0.6, np.inf)))
    #
    #     amp, t2_0, t2_1, freq_h, freq_l, re, base, zz = fit_params
    #     nu_po = (freq_l - freq_h)/2
    #
    #     l = int(len(y_fit) / 2)
    #     fit_x, fit_y = y_fit[:l], y_fit[l:]
    #
    #     fig = Figure()
    #     _ = FigureCanvasSVG(fig)
    #     axs = fig.subplots(2, 1)
    #     data_tx_us = data_tx * 1e6
    #     data_ty_us = data_ty * 1e6
    #
    #     axs[0].errorbar(data_tx_us, data_x, yerr=data_x_err, fmt='bo',
    #              alpha=0.5, capsize=4, markersize=5, label=r'$P(1_1|0_2)$')
    #     axs[0].errorbar(data_ty_us, data_y, yerr=data_y_err, fmt='go',
    #                     alpha=0.5, capsize=4, markersize=5,label=r'$P(1_1|1_2)$')
    #     axs[0].plot(data_tx_us, fit_x, 'b-', data_ty_us, fit_y, 'g-')
    #     axs[0].set_xlabel(r'$Delay [\mu s]$')
    #     axs[0].set_ylabel(r'$P(1)$')
    #     axs[0].legend(loc='upper right', frameon=False)
    #
    #     axs[1].plot(freq_vec_x/1e3, abs(fourier_signal_x), 'b-', freq_vec_y/1e3, abs(fourier_signal_y), 'g-')
    #     axs[1].set_xlim([0, max(freq_vec_x/1e3)])
    #     axs[1].set_xlabel('frequency [kHz]')
    #     axs[1].set_ylabel('abs(FFT)')
    #     axs[1].annotate(r'$\nu/2\pi$' + f' = {round(nu_po/ 1e3, 1)}kHz \n'
    #                     f'zz = {round(zz/ 1e3, 1)}kHz \n'
    #                     f'b = {round(re,3)} \n'
    #                     r'$T_2^0$' + f' = {round(t2_0 / 1e-6, 1)}uS \n'
    #                     r'$T_2^1$' + f' = {round(t2_1 / 1e-6, 1)}uS \n',
    #                     # f'freq_h = {round(freq_h/ 1e3, 1)}kHz \n'
    #                     # f'freq_l = {round(freq_l/ 1e3, 1)}kHz',
    #                     (.7, .5), xycoords='axes fraction')
    #     fig.subplots_adjust(hspace=0.35)
    #
    #     # return analysis_results, figs + [fig]
    #     return [], [fig]

    @classmethod
    def _default_options(cls) -> Options:
        """Return the default analysis options.
        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """

        default_options = super()._default_options()
        default_options.curve_drawer.set_options(
            xlabel="Delay",
            ylabel="P(0)",
            xval_unit="s",
        )
        default_options.result_parameters = [
            curve.ParameterRepr("zz", "zz", "Hz"),
            curve.ParameterRepr("t2_0", "T2(0)_zz", "s"),
            curve.ParameterRepr("t2_1", "T2(1)_zz", "s"),
        ]

        return default_options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """

        user_opt.bounds.set_if_empty(
            amp=(0.3, 0.55),
            t2_0=(7e-6, 250e-6),
            t2_1=(7e-6, 250e-6),
            re=(0.0, 1.0),
            base=(0.3, 0.55),
        )

        # Default guess values
        guesses_dict = dict()
        for series in ["X|0", "X|1"]:
            data = curve_data.get_subset_of(series)
            if self._cut_off_delay is not None:
                fourier_signal, freq_vec = myfft(
                    data.y[: self._cut_off_delay]
                    - np.mean(data.y[: self._cut_off_delay]),
                    dt=(data.x[1] - data.x[0]),
                )
            else:
                fourier_signal, freq_vec = myfft(
                    data.y - np.mean(data.y), dt=(data.x[1] - data.x[0])
                )
            top_freqs = n_dominant_freq(fourier_signal, freq_vec, n=2)
            guesses_dict[series] = {"top_freqs": top_freqs, "zero_delay": data.y[0]}

        zz_guess = (
            (guesses_dict["X|0"]["top_freqs"][0] - guesses_dict["X|1"]["top_freqs"][0])
            + (
                guesses_dict["X|0"]["top_freqs"][1]
                - guesses_dict["X|1"]["top_freqs"][1]
            )
        ) / 4
        freq_e_guess = (
            guesses_dict["X|0"]["top_freqs"][0] + guesses_dict["X|1"]["top_freqs"][0]
        ) / 2
        freq_o_guess = (
            guesses_dict["X|0"]["top_freqs"][1] + guesses_dict["X|1"]["top_freqs"][1]
        ) / 2

        user_opt.p0.set_if_empty(
            t2_0=1e-4,
            t2_1=1e-4,
            re=0.5,
            amp=0.5,
            base=0.5,
        )

        # guess all frequencies signs options
        options = []
        for i in (-1, 1):
            for j in (-1, 1):
                for k in (-1, 1):
                    opt = user_opt.copy()
                    opt.p0.set_if_empty(
                        freq_e=i * freq_e_guess,
                        freq_o=j * freq_o_guess,
                        zz=k * zz_guess,
                    )
                    options.append(opt)

        return options

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.
        A good fit has:
            - a reduced chi-squared lower than three,
            - an error on the frequency smaller than the frequency.
        """
        fit_freq = fit_data.fitval("freq_e")

        criteria = [
            fit_data.reduced_chisq < 3,
        ]

        if all(criteria):
            return "good"

        return "bad"


class ZZParityPlusAnalysis(curve.CurveAnalysis):
    r"""The ZZ Ramsey analysis is based on a fit to a cosine function.
    # section: fit_model
        Analyse a ZZ Ramsey experiment by fitting the '0' and '1' series to cosine
        functions. The two functions share the frequency and amplitude parameters
        (i.e. beta).
        .. math::
            y_0 = {\rm amp} e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq}\cdot x {\rm phase}) + {\rm base} \\
            y_1 = {\rm amp} e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq + zz}\cdot x {\rm phase}\right) + {\rm base}"
    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of both series.
            init_guess: The maximum y value less the minimum y value. 0.5 is also tried.
            bounds: [-2, 2] scaled to the maximum signal value.
        defpar \tau:
            desc: The exponential decay of the curve.
            init_guess: The initial guess is obtained by fitting an exponential to the
                square root of ('0' data)**2 + ('1' data)**2.
            bounds: [0, inf].
        defpar \rm base:
            desc: Base line of both series.
            init_guess: The average of the data. 0.5 is also tried.
            bounds: [-1, 1] scaled to the maximum signal value.
        defpar \rm freq:
            desc: Frequency of both series. This is the parameter of interest.
            init_guess: The frequency with the highest power spectral density.
            bounds: [0, inf].
        defpar \rm phase:
            desc: Common phase offset.
            init_guess: Linearly spaced between the maximum and minimum scanned beta.
            bounds: [-min scan range, max scan range].
    """

    def __init__(self, osc_freq: float = 0, cut_off_delay: Optional[int] = None):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="(amp * exp(-x / t2_0) * ( re * cos(2 * pi * (freq_e - zz) * x) + (1-re) * cos(2 * pi * (freq_o - zz) * x) ) + base)",
                    name="X|0",
                    data_sort_key={"series": "X|0"},
                ),
                lmfit.models.ExpressionModel(
                    expr="(amp * exp(-x / t2_1) * ( re * cos(2 * pi * (freq_e + zz) * x) + (1-re) * cos(2 * pi * (freq_o + zz) * x) ) + base)",
                    name="X|1",
                    data_sort_key={"series": "X|1"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp * ( exp(-x / t2_0) * ( re * cos(2 * pi * (freq_e - zz) * x) + (1-re) * cos(2 * pi * (freq_o - zz) * x)) + exp(-x / t2_1) * ( re * cos(2 * pi * (freq_e + zz) * x) + (1-re) * cos(2 * pi * (freq_o + zz) * x)) )/2 + base",
                    name="X|P",
                    data_sort_key={"series": "X|P"},
                ),
            ]
        )
        self._osc_freq = osc_freq
        self._cut_off_delay = cut_off_delay
        self._options.data_processor = DataProcessor(
            input_key="counts", data_actions=[Probability(outcome="0")]
        )

    def _run_analysis(self, experiment_data):
        analysis_results, figs = super()._run_analysis(experiment_data)
        #
        # freq_e = next(
        #     filter(lambda res: res.name == "freq_e", analysis_results)).value
        # freq_o = next(
        #     filter(lambda res: res.name == "freq_o", analysis_results)).value
        # delta = (freq_e + freq_o) / 2 - self._osc_freq
        # nu = abs((freq_e - freq_o)) / 2
        #
        # parity_result = [
        #     AnalysisResultData(
        #         name="Delta_ZZ",
        #         value=delta,
        #         chisq=None,
        #         quality=None,
        #         extra={"unit": "Hz"}),
        #     AnalysisResultData(
        #         name="nu_ZZ",
        #         value=nu,
        #         chisq=None,
        #         quality=None,
        #         extra={"unit": "Hz"})
        # ]
        # # Return combined results
        # analysis_results = parity_result + analysis_results
        # figs[0].figure.axes[0].set_title(f"Delta={delta}, nu={nu}")
        return analysis_results, figs

    @classmethod
    def _default_options(cls) -> Options:
        """Return the default analysis options.
        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """

        default_options = super()._default_options()
        default_options.curve_drawer.set_options(
            xlabel="Delay",
            ylabel="P(0)",
            xval_unit="s",
        )
        default_options.result_parameters = [
            curve.ParameterRepr("zz", "zz", "Hz"),
            curve.ParameterRepr("t2_0", "T2(0)_zz", "s"),
            curve.ParameterRepr("t2_1", "T2(1)_zz", "s"),
        ]

        return default_options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """

        user_opt.bounds.set_if_empty(
            amp=(0.3, 0.55),
            t2_0=(7e-6, 250e-6),
            t2_1=(7e-6, 250e-6),
            re=(0.0, 1.0),
            base=(0.3, 0.55),
        )

        # Default guess values
        guesses_dict = dict()
        for series in ["X|0", "X|1"]:
            data = curve_data.get_subset_of(series)
            if self._cut_off_delay is not None:
                fourier_signal, freq_vec = myfft(
                    data.y[: self._cut_off_delay]
                    - np.mean(data.y[: self._cut_off_delay]),
                    dt=(data.x[1] - data.x[0]),
                )
            else:
                fourier_signal, freq_vec = myfft(
                    data.y - np.mean(data.y), dt=(data.x[1] - data.x[0])
                )
            top_freqs = n_dominant_freq(fourier_signal, freq_vec, n=2)
            guesses_dict[series] = {"top_freqs": top_freqs, "zero_delay": data.y[0]}

        zz_guess = (
            (guesses_dict["X|0"]["top_freqs"][0] - guesses_dict["X|1"]["top_freqs"][0])
            + (
                guesses_dict["X|0"]["top_freqs"][1]
                - guesses_dict["X|1"]["top_freqs"][1]
            )
        ) / 4
        freq_e_guess = (
            guesses_dict["X|0"]["top_freqs"][0] + guesses_dict["X|1"]["top_freqs"][0]
        ) / 2
        freq_o_guess = (
            guesses_dict["X|0"]["top_freqs"][1] + guesses_dict["X|1"]["top_freqs"][1]
        ) / 2

        user_opt.p0.set_if_empty(
            t2_0=1e-4,
            t2_1=1e-4,
            re=0.5,
            amp=0.5,  # amp_guess,
            base=0.5,  # base_guess,
        )

        # guess all frequencies signs options
        options = []
        for i in (-1, 1):
            for j in (-1, 1):
                for k in (-1, 1):
                    opt = user_opt.copy()
                    opt.p0.set_if_empty(
                        freq_e=i * freq_e_guess,
                        freq_o=j * freq_e_guess,
                        zz=k * zz_guess,
                    )
                    options.append(opt)

        return options

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.
        A good fit has:
            - a reduced chi-squared lower than three,
            - an error on the frequency smaller than the frequency.
        """
        fit_freq = fit_data.fitval("freq_e")

        criteria = [
            fit_data.reduced_chisq < 3,
        ]

        if all(criteria):
            return "good"

        return "bad"


class ZZParityPlus2Analysis(curve.CurveAnalysis):
    r"""The ZZ Ramsey analysis is based on a fit to a cosine function.
    # section: fit_model
        Analyse a ZZ Ramsey experiment by fitting the '0' and '1' series to cosine
        functions. The two functions share the frequency and amplitude parameters
        (i.e. beta).
        .. math::
            y_0 = {\rm amp} e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq}\cdot x {\rm phase}) + {\rm base} \\
            y_1 = {\rm amp} e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq + zz}\cdot x {\rm phase}\right) + {\rm base}"
    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of both series.
            init_guess: The maximum y value less the minimum y value. 0.5 is also tried.
            bounds: [-2, 2] scaled to the maximum signal value.
        defpar \tau:
            desc: The exponential decay of the curve.
            init_guess: The initial guess is obtained by fitting an exponential to the
                square root of ('0' data)**2 + ('1' data)**2.
            bounds: [0, inf].
        defpar \rm base:
            desc: Base line of both series.
            init_guess: The average of the data. 0.5 is also tried.
            bounds: [-1, 1] scaled to the maximum signal value.
        defpar \rm freq:
            desc: Frequency of both series. This is the parameter of interest.
            init_guess: The frequency with the highest power spectral density.
            bounds: [0, inf].
        defpar \rm phase:
            desc: Common phase offset.
            init_guess: Linearly spaced between the maximum and minimum scanned beta.
            bounds: [-min scan range, max scan range].
    """

    def __init__(
        self,
        physical_qubits: Tuple[int],
        osc_freq: float = 0,
        cut_off_delay: Optional[int] = None,
    ):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp0 * ( exp(-x / t2_0_0) * ( 0.5 * cos(2 * pi * (freq_e0 - zz) * x) + 0.5 * cos(2 * pi * (freq_o0 - zz) * x)) + exp(-x / t2_1_0) * ( 0.5 * cos(2 * pi * (freq_e0 + zz) * x) + 0.5 * cos(2 * pi * (freq_o0 + zz) * x)) )/2 + base0",
                    name="X_0",
                    data_sort_key={"series": "X_0"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp0 * ( exp(-x / t2_0_0) * ( 0.5 * sin(2 * pi * (freq_e0 - zz) * x) + 0.5 * sin(2 * pi * (freq_o0 - zz) * x)) + exp(-x / t2_1_0) * ( 0.5 * sin(2 * pi * (freq_e0 + zz) * x) + 0.5 * sin(2 * pi * (freq_o0 + zz) * x)) )/2 + base0",
                    name="Y_0",
                    data_sort_key={"series": "Y_0"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp1 * ( exp(-x / t2_0_1) * ( 0.5 * cos(2 * pi * (freq_e1 - zz) * x) + 0.5 * cos(2 * pi * (freq_o1 - zz) * x)) + exp(-x / t2_1_1) * ( 0.5 * cos(2 * pi * (freq_e1 + zz) * x) + 0.5 * cos(2 * pi * (freq_o1 + zz) * x)) )/2 + base1",
                    name="X_1",
                    data_sort_key={"series": "X_1"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp1 * ( exp(-x / t2_0_1) * ( 0.5 * sin(2 * pi * (freq_e1 - zz) * x) + 0.5 * sin(2 * pi * (freq_o1 - zz) * x)) + exp(-x / t2_1_1) * ( 0.5 * sin(2 * pi * (freq_e1 + zz) * x) + 0.5 * sin(2 * pi * (freq_o1 + zz) * x)) )/2 + base1",
                    name="Y_1",
                    data_sort_key={"series": "Y_1"},
                ),
            ]
        )
        self._osc_freq = osc_freq
        self._cut_off_delay = cut_off_delay
        self._labels = ["X_0", "Y_0", "X_1", "Y_1"]

        # fake model to get the data from circuits
        self._auxiliary_models = [
            lmfit.models.ExpressionModel(
                expr="a*x",
                name="X",
                data_sort_key={"series": "X"},
            ),
            lmfit.models.ExpressionModel(
                expr="a*x",
                name="Y",
                data_sort_key={"series": "Y"},
            ),
        ]
        self.plot_fourier = False
        self._save_data = False
        self._physical_qubits = physical_qubits

    def _run_analysis(self, experiment_data):
        figures, analysis_result = [], []

        data = self._full_data(experiment_data)
        RamX_0 = data.get_subset_of("X_0")
        RamY_0 = data.get_subset_of("Y_0")
        RamX_1 = data.get_subset_of("X_1")
        RamY_1 = data.get_subset_of("Y_1")

        # Run fitting
        fit_data = self._run_curve_fit(
            curve_data=data,
            models=self._models,
        )

        fig = Figure()
        _ = FigureCanvasSVG(fig)
        axs = fig.subplots(2, 1)

        axs[0].errorbar(
            RamX_0.x * 1e6,
            RamX_0.y,
            yerr=RamX_0.y_err,
            fmt="bo",
            alpha=0.5,
            capsize=4,
            markersize=5,
            label=r"$X, q_1$",
        )
        axs[0].errorbar(
            RamY_0.x * 1e6,
            RamY_0.y,
            yerr=RamY_0.y_err,
            fmt="go",
            alpha=0.5,
            capsize=4,
            markersize=5,
            label=r"$Y, q_1$",
        )

        axs[0].set_xlabel(r"$Delay [\mu s]$")
        axs[0].set_ylabel(r"$P(0)$")
        axs[0].legend(loc="upper right", frameon=False)
        axs[0].set_title(f"Driven freq = {round(self._osc_freq / 1e3, 1)}kHz")

        axs[1].errorbar(
            RamX_1.x * 1e6,
            RamX_1.y,
            yerr=RamX_1.y_err,
            fmt="bo",
            alpha=0.5,
            capsize=4,
            markersize=5,
            label=r"$X, q_2$",
        )

        axs[1].errorbar(
            RamY_1.x * 1e6,
            RamY_1.y,
            yerr=RamY_1.y_err,
            fmt="go",
            alpha=0.5,
            capsize=4,
            markersize=5,
            label=r"$Y, q_2$",
        )

        axs[1].set_xlabel(r"$Delay [\mu s]$")
        axs[1].set_ylabel(r"$P(0)$")
        axs[1].legend(loc="upper right", frameon=False)

        interp_x = np.linspace(np.min(RamX_0.x), np.max(RamX_0.x), num=300)
        if fit_data.success:
            RamX_0_fit_with_err = eval_with_uncertainties(
                x=interp_x,
                model=self._models[0],
                params=fit_data.ufloat_params,
            )
            RamY_0_fit_with_err = eval_with_uncertainties(
                x=interp_x,
                model=self._models[1],
                params=fit_data.ufloat_params,
            )
            RamX_1_fit_with_err = eval_with_uncertainties(
                x=interp_x,
                model=self._models[2],
                params=fit_data.ufloat_params,
            )
            RamY_1_fit_with_err = eval_with_uncertainties(
                x=interp_x,
                model=self._models[3],
                params=fit_data.ufloat_params,
            )
            axs[0].plot(
                interp_x * 1e6,
                unp.nominal_values(RamX_0_fit_with_err),
                color="blue",
                linestyle="-",
            )
            axs[0].plot(
                interp_x * 1e6,
                unp.nominal_values(RamY_0_fit_with_err),
                color="green",
                linestyle="-",
            )
            axs[1].plot(
                interp_x * 1e6,
                unp.nominal_values(RamX_1_fit_with_err),
                color="blue",
                linestyle="-",
            )
            axs[1].plot(
                interp_x * 1e6,
                unp.nominal_values(RamY_1_fit_with_err),
                color="green",
                linestyle="-",
            )

            zz = fit_data.ufloat_params["zz"]
            nu_1 = (
                fit_data.ufloat_params["freq_o0"] - fit_data.ufloat_params["freq_e0"]
            ) / 2
            nu_2 = (
                fit_data.ufloat_params["freq_o1"] - fit_data.ufloat_params["freq_e1"]
            ) / 2
            t2_0_q1 = fit_data.ufloat_params["t2_0_0"]
            t2_1_q1 = fit_data.ufloat_params["t2_1_0"]
            t2_0_q2 = fit_data.ufloat_params["t2_0_1"]
            t2_1_q2 = fit_data.ufloat_params["t2_1_1"]

            if zz.n < 0:
                zz = -zz
                t2_0_q1, t2_1_q1 = t2_1_q1, t2_0_q1
                t2_0_q2, t2_1_q2 = t2_1_q2, t2_0_q2

            Delta_1 = (
                (fit_data.ufloat_params["freq_o0"] + fit_data.ufloat_params["freq_e0"])
                / 2
                - self._osc_freq
                - zz
            )
            Delta_2 = (
                (fit_data.ufloat_params["freq_o1"] + fit_data.ufloat_params["freq_e1"])
                / 2
                - self._osc_freq
                - zz
            )

            analysis_result += [
                AnalysisResultData(
                    name="ZZ",
                    value=zz,
                    chisq=fit_data.reduced_chisq,
                    quality=None,
                    extra={"unit": "Hz"},
                ),
                AnalysisResultData(
                    name="nu",
                    value=nu_1,
                    chisq=fit_data.reduced_chisq,
                    quality=None,
                    extra={"unit": "Hz"},
                    device_components=[Qubit(self._physical_qubits[0])],
                ),
                AnalysisResultData(
                    name="nu",
                    value=nu_2,
                    chisq=fit_data.reduced_chisq,
                    quality=None,
                    extra={"unit": "Hz"},
                    device_components=[Qubit(self._physical_qubits[1])],
                ),
                AnalysisResultData(
                    name="T2",
                    value=t2_0_q1,
                    chisq=fit_data.reduced_chisq,
                    quality=None,
                    extra={"unit": "s"},
                    device_components=[Qubit(self._physical_qubits[0])],
                ),
                AnalysisResultData(
                    name="T2_1",
                    value=t2_1_q1,
                    chisq=fit_data.reduced_chisq,
                    quality=None,
                    extra={"unit": "s"},
                    device_components=[Qubit(self._physical_qubits[0])],
                ),
                AnalysisResultData(
                    name="T2",
                    value=t2_0_q2,
                    chisq=fit_data.reduced_chisq,
                    quality=None,
                    extra={"unit": "s"},
                    device_components=[Qubit(self._physical_qubits[1])],
                ),
                AnalysisResultData(
                    name="T2_1",
                    value=t2_1_q2,
                    chisq=fit_data.reduced_chisq,
                    quality=None,
                    extra={"unit": "s"},
                    device_components=[Qubit(self._physical_qubits[1])],
                ),
                AnalysisResultData(
                    name="Delta",
                    value=Delta_1,
                    chisq=fit_data.reduced_chisq,
                    quality=None,
                    extra={"unit": "Hz"},
                    device_components=[Qubit(self._physical_qubits[0])],
                ),
                AnalysisResultData(
                    name="Delta",
                    value=Delta_2,
                    chisq=fit_data.reduced_chisq,
                    quality=None,
                    extra={"unit": "Hz"},
                    device_components=[Qubit(self._physical_qubits[1])],
                ),
            ]

            axs[0].annotate(
                f"zz = {round(zz.n/ 1e3, 1)}kHz \n"
                r"$\nu_1$" + f" = {round(nu_1.n/ 1e3, 1)}kHz \n"
                r"$T_2^{(0)}$" + f" = {round(t2_0_q1.n / 1e-6, 1)}uS \n"
                r"$T_2^{(1)}$" + f" = {round(t2_1_q1.n / 1e-6, 1)}uS \n",
                (0.5, 0.45),
                xycoords="axes fraction",
            )
            axs[0].annotate(
                r"$\Delta_1$" + f" = {round(Delta_1.n / 1e3, 1)}kHz \n",
                (0.2, 0.8),
                xycoords="axes fraction",
            )
            axs[1].annotate(
                r"$\nu_2$" + f" = {round(nu_2.n / 1e3, 1)}kHz \n"
                r"$T_2^{(0)}$" + f" = {round(t2_0_q2.n / 1e-6, 1)}uS \n"
                r"$T_2^{(1)}$" + f" = {round(t2_1_q2.n / 1e-6, 1)}uS \n"
                r"$reduced-\chi^2$" + f" = {round(fit_data.reduced_chisq,2)}",
                (0.5, 0.55),
                xycoords="axes fraction",
            )
            axs[1].annotate(
                r"$\Delta_2$" + f" = {round(Delta_2.n / 1e3, 1)}kHz \n",
                (0.2, 0.8),
                xycoords="axes fraction",
            )

            #  ----------  addition plot ------------------
            additional_plot = False
            if additional_plot:
                fig2 = Figure()
                _ = FigureCanvasSVG(fig)
                axs2 = fig2.subplots(2, 1)

                axs2[0].errorbar(
                    RamX_0.x * 1e6,
                    2 * RamX_0.y - 1,
                    yerr=2 * RamX_0.y_err,
                    fmt="bo",
                    alpha=0.5,
                    capsize=4,
                    markersize=5,
                    label=r"$X, q_1$",
                )
                axs2[1].errorbar(
                    RamY_0.x * 1e6,
                    2 * RamY_0.y - 1,
                    yerr=2 * RamY_0.y_err,
                    fmt="bo",
                    alpha=0.5,
                    capsize=4,
                    markersize=5,
                    label=r"$Y, q_1$",
                )

                axs2[0].set_xlabel(r"$Delay [\mu s]$")
                axs2[0].set_ylabel(r"$<X>$")

                axs2[0].set_title(f"Driven freq = {round(self._osc_freq / 1e3, 1)}kHz")

                axs2[0].errorbar(
                    RamX_1.x * 1e6,
                    2 * RamX_1.y - 1,
                    yerr=2 * RamX_1.y_err,
                    fmt="ro",
                    alpha=0.5,
                    capsize=4,
                    markersize=5,
                    label=r"$X, q_2$",
                )

                axs2[1].errorbar(
                    RamY_1.x * 1e6,
                    2 * RamY_1.y - 1,
                    yerr=2 * RamY_1.y_err,
                    fmt="ro",
                    alpha=0.5,
                    capsize=4,
                    markersize=5,
                    label=r"$Y, q_2$",
                )

                axs2[1].set_xlabel(r"$Delay [\mu s]$")
                axs2[1].set_ylabel(r"$<Y>$")

                axs2[0].legend(loc="upper right", frameon=False)
                axs2[1].legend(loc="upper right", frameon=False)

                axs2[0].plot(
                    interp_x * 1e6,
                    2 * unp.nominal_values(RamX_0_fit_with_err) - 1,
                    color="blue",
                    linestyle="-",
                )
                axs2[1].plot(
                    interp_x * 1e6,
                    2 * unp.nominal_values(RamY_0_fit_with_err) - 1,
                    color="blue",
                    linestyle="-",
                )
                axs2[0].plot(
                    interp_x * 1e6,
                    2 * unp.nominal_values(RamX_1_fit_with_err) - 1,
                    color="red",
                    linestyle="-",
                )
                axs2[1].plot(
                    interp_x * 1e6,
                    2 * unp.nominal_values(RamY_1_fit_with_err) - 1,
                    color="red",
                    linestyle="-",
                )
                axs2[0].annotate(
                    f"zz = {round(zz.n / 1e3, 1)}kHz \n"
                    r"$\nu_1$" + f" = {round(nu_1.n / 1e3, 1)}kHz \n"
                    r"$T_2^{(0)}$" + f" = {round(t2_0_q1.n / 1e-6, 1)}uS \n"
                    r"$T_2^{(1)}$" + f" = {round(t2_1_q1.n / 1e-6, 1)}uS \n",
                    (0.5, 0.45),
                    xycoords="axes fraction",
                )
                axs2[0].annotate(
                    r"$\Delta_1$" + f" = {round(Delta_1.n / 1e3, 1)}kHz \n",
                    (0.2, 0.8),
                    xycoords="axes fraction",
                )
                axs2[1].annotate(
                    r"$\nu_2$" + f" = {round(nu_2.n / 1e3, 1)}kHz \n"
                    r"$T_2^{(0)}$" + f" = {round(t2_0_q2.n / 1e-6, 1)}uS \n"
                    r"$T_2^{(1)}$" + f" = {round(t2_1_q2.n / 1e-6, 1)}uS \n"
                    r"$reduced-\chi^2$" + f" = {round(fit_data.reduced_chisq, 2)}",
                    (0.5, 0.55),
                    xycoords="axes fraction",
                )
                axs2[1].annotate(
                    r"$\Delta_2$" + f" = {round(Delta_2.n / 1e3, 1)}kHz \n",
                    (0.2, 0.8),
                    xycoords="axes fraction",
                )

                fig2.subplots_adjust(hspace=0.35)
                figures.append(fig2)

        fig.subplots_adjust(hspace=0.35)
        figures.append(fig)

        # --------- Fourier Plot -----------
        if self.plot_fourier:
            if self._cut_off_delay is not None:
                fourier_signal_0x, freq_vec_0x = myfft(
                    RamX_0.y[: self._cut_off_delay]
                    - np.mean(RamX_0.y[: self._cut_off_delay]),
                    RamX_0.x[1] - RamX_0.x[0],
                )
                fourier_signal_1x, freq_vec_1x = myfft(
                    RamX_1.y[: self._cut_off_delay]
                    - np.mean(RamX_1.y[: self._cut_off_delay]),
                    RamX_1.x[1] - RamX_1.x[0],
                )
            else:
                fourier_signal_0x, freq_vec_0x = myfft(
                    RamX_0.y - np.mean(RamX_0.y), RamX_0.x[1] - RamX_0.x[0]
                )
                fourier_signal_1x, freq_vec_1x = myfft(
                    RamX_1.y - np.mean(RamX_1.y), RamX_1.x[1] - RamX_1.x[0]
                )
            fig_fourier = Figure()
            _ = FigureCanvasSVG(fig_fourier)
            ax = fig_fourier.subplots(1, 1)
            ax.plot(freq_vec_0x / 1e3, abs(fourier_signal_0x), label="q1-X")
            ax.plot(freq_vec_1x / 1e3, abs(fourier_signal_1x), label="q2-X")
            ax.set_xlim([0, max(freq_vec_0x / 1e3)])
            ax.set_xlabel("frequency [kHz]")
            ax.set_ylabel("abs(FFT)")
            ax.legend()
            figures.append(fig_fourier)
        # ------------------------------------

        return analysis_result, figures

    @classmethod
    def _default_options(cls) -> Options:
        """Return the default analysis options.
        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """

        default_options = super()._default_options()
        default_options.curve_drawer.set_options(
            xlabel="Delay",
            ylabel="P(0)",
            xval_unit="s",
        )
        default_options.result_parameters = [
            curve.ParameterRepr("zz", "zz", "Hz"),
            curve.ParameterRepr("t2_0", "T2(0)_zz", "s"),
            curve.ParameterRepr("t2_1", "T2(1)_zz", "s"),
        ]

        return default_options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """

        # Default guess values
        guesses_dict = dict()
        for series in ["X_0", "Y_0", "X_1", "Y_1"]:
            data = curve_data.get_subset_of(series)
            if self._cut_off_delay is not None:
                fourier_signal, freq_vec = myfft(
                    data.y[: self._cut_off_delay]
                    - np.mean(data.y[: self._cut_off_delay]),
                    dt=(data.x[1] - data.x[0]),
                )
            else:
                fourier_signal, freq_vec = myfft(
                    data.y - np.mean(data.y), dt=(data.x[1] - data.x[0])
                )
            top_freqs = n_dominant_freq(fourier_signal, freq_vec, n=4)
            guesses_dict[series] = {"top_freqs": top_freqs, "zero_delay": data.y[0]}

        amp0_guess = (
            guesses_dict["X_0"]["zero_delay"] - guesses_dict["Y_0"]["zero_delay"]
        )
        amp1_guess = (
            guesses_dict["X_1"]["zero_delay"] - guesses_dict["Y_1"]["zero_delay"]
        )
        base0_guess = guesses_dict["Y_0"]["zero_delay"]
        base1_guess = guesses_dict["Y_1"]["zero_delay"]

        freq_e0_guess = (
            guesses_dict["X_0"]["top_freqs"][0] + guesses_dict["X_0"]["top_freqs"][2]
        ) / 2
        freq_o0_guess = (
            guesses_dict["X_0"]["top_freqs"][1] + guesses_dict["X_0"]["top_freqs"][3]
        ) / 2

        freq_e1_guess = (
            guesses_dict["X_1"]["top_freqs"][0] + guesses_dict["X_1"]["top_freqs"][2]
        ) / 2
        freq_o1_guess = (
            guesses_dict["X_1"]["top_freqs"][1] + guesses_dict["X_1"]["top_freqs"][3]
        ) / 2

        zz_guess = (
            guesses_dict["X_0"]["top_freqs"][0]
            + guesses_dict["X_0"]["top_freqs"][1]
            - guesses_dict["X_0"]["top_freqs"][2]
            - guesses_dict["X_0"]["top_freqs"][3]
        ) / 4

        user_opt.bounds.set_if_empty(
            amp0=(amp0_guess - 0.08, amp0_guess + 0.08),
            amp1=(amp0_guess - 0.08, amp0_guess + 0.08),
            t2_0_0=(7e-6, 250e-6),
            t2_0_1=(7e-6, 250e-6),
            t2_1_0=(7e-6, 250e-6),
            t2_1_1=(7e-6, 250e-6),
            base0=(base0_guess - 0.08, base0_guess + 0.08),
            base1=(base1_guess - 0.08, base1_guess + 0.08),
        )
        user_opt.p0.set_if_empty(
            t2_0_0=1e-4,
            t2_1_0=1e-4,
            t2_0_1=1e-4,
            t2_1_1=1e-4,
            amp0=amp0_guess,
            amp1=amp1_guess,
            base0=base0_guess,
            base1=base1_guess,
        )

        # guess all frequencies signs options
        options = []
        for i in (-1, 1):
            for j in (-1, 1):
                for k in (-1, 1):
                    opt = user_opt.copy()
                    opt.p0.set_if_empty(
                        freq_o0=i * freq_o0_guess, freq_e0=i * freq_e0_guess
                    )
                    opt.p0.set_if_empty(
                        freq_o1=j * freq_o1_guess, freq_e1=j * freq_e1_guess
                    )
                    opt.p0.set_if_empty(zz=k * zz_guess)
                    options.append(opt)

        return options

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.
        A good fit has:
            - a reduced chi-squared lower than three,
            - an error on the frequency smaller than the frequency.
        """
        fit_freq = fit_data.fitval("freq_e1")

        criteria = [
            fit_data.reduced_chisq < 3,
        ]

        if all(criteria):
            return "good"

        return "bad"

    def _full_data(self, experiment_data) -> CurveData:
        formatted_data_list = []
        qubits = [0, 1]
        for i in qubits:
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
            formatted_data_list.append(self._format_data(processed_data))

        formatted_data = combine_curve_data(formatted_data_list, self._labels)

        if self._save_data:
            import pickle
            from datetime import datetime

            t = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            with open(
                r"C:\Users\018280756\Documents\data\data_" + t + "ZZ2.pickle", "wb"
            ) as handle:
                pickle.dump(formatted_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

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


class ZZParityPlus3Analysis(curve.CurveAnalysis):
    r"""The ZZ Ramsey analysis is based on a fit to a cosine function.
    # section: fit_model
        Analyse a ZZ Ramsey experiment by fitting the '0' and '1' series to cosine
        functions. The two functions share the frequency and amplitude parameters
        (i.e. beta).
        .. math::
            y_0 = {\rm amp} e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq}\cdot x {\rm phase}) + {\rm base} \\
            y_1 = {\rm amp} e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq + zz}\cdot x {\rm phase}\right) + {\rm base}"
    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of both series.
            init_guess: The maximum y value less the minimum y value. 0.5 is also tried.
            bounds: [-2, 2] scaled to the maximum signal value.
        defpar \tau:
            desc: The exponential decay of the curve.
            init_guess: The initial guess is obtained by fitting an exponential to the
                square root of ('0' data)**2 + ('1' data)**2.
            bounds: [0, inf].
        defpar \rm base:
            desc: Base line of both series.
            init_guess: The average of the data. 0.5 is also tried.
            bounds: [-1, 1] scaled to the maximum signal value.
        defpar \rm freq:
            desc: Frequency of both series. This is the parameter of interest.
            init_guess: The frequency with the highest power spectral density.
            bounds: [0, inf].
        defpar \rm phase:
            desc: Common phase offset.
            init_guess: Linearly spaced between the maximum and minimum scanned beta.
            bounds: [-min scan range, max scan range].
    """

    def __init__(self, osc_freq: float = 0, cut_off_delay: Optional[int] = None):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp0 * ("
                    "exp(-x / t2_q0_0) * "
                    "(cos(2 * pi * (freq_e0 - zz1) * x) + cos(2 * pi * (freq_o0 - zz1) * x))"
                    "+ exp(-x / t2_q0_1) *"
                    "(cos(2 * pi * (freq_e0 + zz1) * x) + cos(2 * pi * (freq_o0 + zz1) * x)) "
                    ")/4 + base0",
                    name="X_0",
                    data_sort_key={"series": "X_0"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp0 * ("
                    "exp(-x / t2_q0_0) * "
                    "(sin(2 * pi * (freq_e0 - zz1) * x) + sin(2 * pi * (freq_o0 - zz1) * x))"
                    "+ exp(-x / t2_q0_1) *"
                    "(sin(2 * pi * (freq_e0 + zz1) * x) + sin(2 * pi * (freq_o0 + zz1) * x))"
                    ")/4 + base0",
                    name="Y_0",
                    data_sort_key={"series": "Y_0"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp1 * ("
                    "exp(-x / t2_q1_00) * "
                    "(cos(2 * pi * (freq_e1 - zz1 - zz2) * x) + cos(2 * pi * (freq_o1 - zz1 - zz2) * x))"
                    "+ exp(-x / t2_q1_10) * "
                    "(cos(2 * pi * (freq_e1 + zz1 - zz2) * x) + cos(2 * pi * (freq_o1 + zz1 - zz2) * x))"
                    "+ exp(-x / t2_q1_01) *"
                    "(cos(2 * pi * (freq_e1 - zz1 + zz2) * x) + cos(2 * pi * (freq_o1 - zz1 + zz2) * x))"
                    "+ exp(-x / t2_q1_11) *"
                    "(cos(2 * pi * (freq_e1 + zz1 + zz2) * x) + cos(2 * pi * (freq_o1 + zz1 + zz2) * x))"
                    ")/8 + base1",
                    name="X_1",
                    data_sort_key={"series": "X_1"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp1 * ("
                    "exp(-x / t2_q1_00) * "
                    "(sin(2 * pi * (freq_e1 - zz1 - zz2) * x) + sin(2 * pi * (freq_o1 - zz1 - zz2) * x))"
                    "+ exp(-x / t2_q1_10) * "
                    "(sin(2 * pi * (freq_e1 + zz1 - zz2) * x) + sin(2 * pi * (freq_o1 + zz1 - zz2) * x))"
                    "+ exp(-x / t2_q1_01) *"
                    "(sin(2 * pi * (freq_e1 - zz1 + zz2) * x) + sin(2 * pi * (freq_o1 - zz1 + zz2) * x))"
                    "+ exp(-x / t2_q1_11) *"
                    "(sin(2 * pi * (freq_e1 + zz1 + zz2) * x) + sin(2 * pi * (freq_o1 + zz1 + zz2) * x))"
                    ")/8 + base1",
                    name="Y_1",
                    data_sort_key={"series": "Y_1"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp2 * ("
                    "exp(-x / t2_q2_0) * "
                    "(cos(2 * pi * (freq_e2 - zz2) * x) + cos(2 * pi * (freq_o2 - zz2) * x))"
                    "+ exp(-x / t2_q2_1) *"
                    "(cos(2 * pi * (freq_e0 + zz2) * x) + cos(2 * pi * (freq_o2 + zz2) * x))"
                    ")/4 + base2",
                    name="X_2",
                    data_sort_key={"series": "X_2"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp2 * ("
                    "exp(-x / t2_q2_0) * "
                    "(sin(2 * pi * (freq_e2 - zz2) * x) + sin(2 * pi * (freq_o2 - zz2) * x))"
                    "+ exp(-x / t2_q0_1) *"
                    "(sin(2 * pi * (freq_e2 + zz2) * x) + sin(2 * pi * (freq_o2 + zz2) * x)) "
                    ")/4 + base2",
                    name="Y_2",
                    data_sort_key={"series": "Y_2"},
                ),
            ]
        )
        self._osc_freq = osc_freq
        self._cut_off_delay = cut_off_delay
        self._labels = ["X_0", "Y_0", "X_1", "Y_1", "X_2", "Y_2"]

        # fake model to get the data from circuits
        self._auxiliary_models = [
            lmfit.models.ExpressionModel(
                expr="a*x",
                name="X",
                data_sort_key={"series": "X"},
            ),
            lmfit.models.ExpressionModel(
                expr="a*x",
                name="Y",
                data_sort_key={"series": "Y"},
            ),
        ]
        self.plot_fourier = False
        self._save_data = False

    def _run_analysis(self, experiment_data):
        figures = []

        data = self._full_data(experiment_data)

        fig = Figure()
        _ = FigureCanvasSVG(fig)
        axs = fig.subplots(3, 1, sharex=True)
        for q in range(3):
            RamX, RamY = data.get_subset_of(f"X_{q}"), data.get_subset_of(f"Y_{q}")
            axs[q].errorbar(
                RamX.x * 1e6,
                RamX.y,
                yerr=RamX.y_err,
                fmt="bo",
                alpha=0.5,
                capsize=4,
                markersize=5,
                label=f"X q_{q+1}",
            )
            axs[q].errorbar(
                RamY.x * 1e6,
                RamY.y,
                yerr=RamY.y_err,
                fmt="go",
                alpha=0.5,
                capsize=4,
                markersize=5,
                label=f"Y q_{q+1}",
            )
            axs[q].set_ylabel(r"$P(0)$")
            axs[q].legend(loc="upper right", frameon=False)
        axs[2].set_xlabel(r"$Delay [\mu s]$")
        axs[0].set_title(f"Driven freq = {round(self._osc_freq / 1e3, 1)}kHz")

        # Run fitting
        fit_data = self._run_curve_fit(
            curve_data=data,
            models=self._models,
        )

        interp_x = np.linspace(np.min(RamX.x), np.max(RamX.x), num=300)
        if fit_data.success:
            for i in range(0, len(self._labels), 2):
                RamX_fit_with_err = eval_with_uncertainties(
                    x=interp_x,
                    model=self._models[i],
                    params=fit_data.ufloat_params,
                )
                RamY_fit_with_err = eval_with_uncertainties(
                    x=interp_x,
                    model=self._models[i + 1],
                    params=fit_data.ufloat_params,
                )
                axs[0].plot(
                    interp_x * 1e6,
                    unp.nominal_values(RamX_fit_with_err),
                    color="blue",
                    linestyle="-",
                )
                axs[0].plot(
                    interp_x * 1e6,
                    unp.nominal_values(RamY_fit_with_err),
                    color="green",
                    linestyle="-",
                )

        #     zz = fit_data.ufloat_params['zz'].n
        #     nu_1 = fit_data.ufloat_params['freq_o0'].n - fit_data.ufloat_params['freq_e0'].n
        #     nu_2 = fit_data.ufloat_params['freq_o1'].n - fit_data.ufloat_params['freq_e1'].n
        #     t2_0_q1 = fit_data.ufloat_params['t2_0_0'].n
        #     t2_1_q1 = fit_data.ufloat_params['t2_1_0'].n
        #     t2_0_q2 = fit_data.ufloat_params['t2_0_1'].n
        #     t2_1_q2 = fit_data.ufloat_params['t2_1_1'].n
        #
        #     axs[0].annotate(f'zz = {round(zz/ 1e3, 1)}kHz \n'
        #                     r'$\nu_1$' + f' = {round(nu_1/ 1e3, 1)}kHz \n'
        #                     r'$T_2^{(0)}$' + f' = {round(t2_0_q1 / 1e-6, 1)}uS \n'
        #                     r'$T_2^{(1)}$' + f' = {round(t2_1_q1 / 1e-6, 1)}uS \n',
        #                     (.5, .4), xycoords='axes fraction')
        #     axs[1].annotate(r'$\nu_2$' + f' = {round(nu_2 / 1e3, 1)}kHz \n'
        #                     r'$T_2^{(0)}$' + f' = {round(t2_0_q2 / 1e-6, 1)}uS \n'
        #                     r'$T_2^{(1)}$' + f' = {round(t2_1_q2 / 1e-6, 1)}uS \n'
        #                     r'$reduce-\chi^2$' + f' = {fit_data.reduced_chisq}',
        #                     (.5, .4), xycoords='axes fraction')

        figures.append(fig)

        # --------- Fourier Plot -----------
        if self.plot_fourier:
            fourier_data = self._fourier_data(data=data)
            fig_fourier = Figure()
            _ = FigureCanvasSVG(fig_fourier)
            ax = fig_fourier.subplots(1, 1)
            for series in ["X_0", "X_1", "X_2"]:
                ax.plot(
                    fourier_data[series]["freq_vec"] / 1e3,
                    abs(fourier_data[series]["fourier_signal"]),
                    label=series,
                )
                ax.set_xlim([0, max(fourier_data[series]["freq_vec"] / 1e3)])
            ax.set_xlabel("frequency [kHz]")
            ax.set_ylabel("abs(FFT)")
            ax.legend()
            figures.append(fig_fourier)
        # ------------------------------------

        return [], figures

    @classmethod
    def _default_options(cls) -> Options:
        """Return the default analysis options.
        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """

        default_options = super()._default_options()
        default_options.curve_drawer.set_options(
            xlabel="Delay",
            ylabel="P(0)",
            xval_unit="s",
        )
        default_options.result_parameters = [
            curve.ParameterRepr("zz", "zz", "Hz"),
            curve.ParameterRepr("t2_0", "T2(0)_zz", "s"),
            curve.ParameterRepr("t2_1", "T2(1)_zz", "s"),
        ]

        return default_options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        fourier_data = self._fourier_data(data=curve_data)
        # Default guess values
        gd = dict()
        for series in self._labels:
            top_freqs = n_dominant_freq(
                fourier_data[series]["fourier_signal"],
                fourier_data[series]["freq_vec"],
                n=4,
            )
            gd[series] = {
                "top_freqs": top_freqs,
                "zero_delay": curve_data.get_subset_of(series).y[0],
            }

        for q in range(3):
            gd[f"amp{q}"] = gd[f"X_{q}"]["zero_delay"] - gd[f"Y_{q}"]["zero_delay"]
            gd[f"base{q}"] = gd[f"Y_{q}"]["zero_delay"]
            gd[f"freq_e{q}"] = (
                gd[f"X_{q}"]["top_freqs"][0] + gd[f"X_{q}"]["top_freqs"][2]
            ) / 2
            gd[f"freq_o{q}"] = (
                gd[f"X_{q}"]["top_freqs"][1] + gd[f"X_{q}"]["top_freqs"][3]
            ) / 2

        gd["zz1"] = (
            gd["X_0"]["top_freqs"][0]
            + gd["X_0"]["top_freqs"][1]
            - gd["X_0"]["top_freqs"][2]
            - gd["X_0"]["top_freqs"][3]
        ) / 4
        gd["zz2"] = (
            gd["X_2"]["top_freqs"][0]
            + gd["X_2"]["top_freqs"][1]
            - gd["X_2"]["top_freqs"][2]
            - gd["X_2"]["top_freqs"][3]
        ) / 4

        for q in range(3):
            user_opt.bounds.set_if_empty(
                **{
                    f"amp{q}": (gd[f"amp{q}"] - 0.08, gd[f"amp{q}"] + 0.08),
                    f"base{q}": (gd[f"base{q}"] - 0.08, gd[f"base{q}"] + 0.08),
                }
            )
            user_opt.p0.set_if_empty(
                **{
                    f"amp{q}": gd[f"amp{q}"],
                    f"base{q}": gd[f"base{q}"],
                }
            )

        user_opt.bounds.set_if_empty(
            t2_q0_0=(7e-6, 250e-6),
            t2_q0_1=(7e-6, 100e-6),
            t2_q1_00=(7e-6, 250e-6),
            t2_q1_10=(7e-6, 100e-6),
            t2_q1_01=(7e-6, 100e-6),
            t2_q1_11=(7e-6, 50e-6),
            t2_q2_0=(7e-6, 250e-6),
            t2_q2_1=(7e-6, 1000e-6),
        )
        user_opt.p0.set_if_empty(
            t2_q0_0=100e-6,
            t2_q0_1=50e-6,
            t2_q1_00=100e-6,
            t2_q1_01=50e-6,
            t2_q1_10=50e-6,
            t2_q1_11=20e-6,
            t2_q2_0=100e-6,
            t2_q2_1=50e-6,
        )

        # guess all frequencies signs options
        options = []
        for i in (-1, 1):
            for j in (-1, 1):
                for k in (-1, 1):
                    opt = user_opt.copy()
                    opt.p0.set_if_empty(
                        freq_o0=i * gd["freq_e0"], freq_e0=i * gd["freq_e0"]
                    )
                    opt.p0.set_if_empty(
                        freq_o1=j * gd["freq_e1"], freq_e1=j * gd["freq_e1"]
                    )
                    opt.p0.set_if_empty(
                        freq_o1=j * gd["freq_e1"], freq_e1=j * gd["freq_e1"]
                    )
                    opt.p0.set_if_empty(zz1=k * gd["zz1"])
                    opt.p0.set_if_empty(zz2=k * gd["zz2"])
                    options.append(opt)

        return options

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.
        A good fit has:
            - a reduced chi-squared lower than three,
            - an error on the frequency smaller than the frequency.
        """
        fit_freq = fit_data.fitval("freq_e1")

        criteria = [
            fit_data.reduced_chisq < 3,
        ]

        if all(criteria):
            return "good"

        return "bad"

    def _full_data(self, experiment_data) -> CurveData:
        formatted_data_list = []
        qubits = [0, 1, 2]
        for i in qubits:
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
            formatted_data_list.append(self._format_data(processed_data))

        formatted_data = combine_curve_data(formatted_data_list, self._labels)

        if self._save_data:
            import pickle
            from datetime import datetime

            t = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            with open(
                r"C:\Users\018280756\Documents\data\data_" + t + "ZZ3.pickle", "wb"
            ) as handle:
                pickle.dump(formatted_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
