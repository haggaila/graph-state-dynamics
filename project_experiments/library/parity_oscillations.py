# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The analysis class for the Parity Oscillations experiment."""
from typing import List, Union, Optional, Sequence

import lmfit
import numpy as np
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure

from project_experiments.library.experiment_utils import myfft, n_dominant_freq
from qiskit_experiments.data_processing import DataProcessor, Probability
from uncertainties import unumpy as unp, UFloat

from qiskit_experiments.framework import AnalysisResultData
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.utils import (
    analysis_result_to_repr,
    eval_with_uncertainties,
)
from project_experiments.library.ramsey_xy import RamseyXY
from qiskit.providers.backend import Backend
from qiskit_experiments.framework.restless_mixin import RestlessMixin

PARAMS_ENTRY_PREFIX = "@Parameters_"


class ParityOscillations(RamseyXY, RestlessMixin):
    r"""Parity Oscillations experiment to measure the charge-parity oscillation frequency and the detuning frequency.

    # section: overview

        This experiment is RamseyXY with longer delay to observe parity oscillations. The analysis includes two
        frequencies: detuning frequency and parity oscillations frequency

    # section: analysis_ref
        :py:class:`ParityOscillationsAnalysis`
    """

    @classmethod
    def _default_experiment_options(cls):
        """Default values for the Parity Oscillation experiment.

        Experiment Options:
            delays (list): The list of delays that will be scanned in the experiment, in seconds.
            osc_freq (float): A frequency shift in Hz that will be applied by means of
                a virtual Z rotation to increase the frequency of the measured oscillation.
        """
        options = super()._default_experiment_options()
        options.delays = np.linspace(0, 50e-6, 50)
        options.osc_freq = 200e3

        return options

    def __init__(
        self,
        qubit: int,
        backend: Optional[Backend] = None,
        delays: Optional[Sequence] = None,
        osc_freq: float = 2e5,
        cut_off_delay: Optional[int] = None,
        fixed_b=True,
        fixed_phase=True,
        file_path_prefix="",
        b_add_z_measurement=False,
        parameters_suffix="",
        b_kappa=False,
    ):
        """Create new experiment.

        Args:
            qubit: The qubit on which to run the Parity Oscillations experiment.
            backend: Optional, the backend to run the experiment on.
            delays: The delays to scan, in seconds.
            osc_freq: The oscillation frequency induced by the user through a virtual
                Rz rotation. This quantity is given in Hz.
            cut_off_delay: The cut-off index of the delay. This argument limits the delays for the fourier analysis.
            fixed_b: fix equal probability of even/odd parity in the fitter
            fixed_phase: fix initial phase of zero in the fitter
            file_path_prefix: path to save the data
            b_add_z_measurement: measure along the z axis, in addition to the ramseyXY experiment
            parameters_suffix: string which added in the end of analysis results parameters
            b_kappa: Adding Gaussian decay envelope in the fitter

        """
        super().__init__(
            qubit,
            backend=backend,
            delays=delays,
            osc_freq=osc_freq,
            b_add_z_measurement=b_add_z_measurement,
        )
        self.analysis = ParityOscillationsAnalysis(
            osc_freq=osc_freq,
            cut_off_delay=cut_off_delay,
            fixed_b=fixed_b,
            fixed_phase=fixed_phase,
            physical_qubit=qubit,
            file_path_prefix=file_path_prefix,
            b_add_z_measurement=b_add_z_measurement,
            parameters_suffix=parameters_suffix,
            b_kappa=b_kappa,
        )


class ParityOscillationsAnalysis(curve.CurveAnalysis):
    r"""Parity Oscillations analysis class.

    # section: fit_model

        Analyse a Parity Oscillations experiment by fitting the X and Y series.
    """

    def __init__(
        self,
        osc_freq: float = 0,
        cut_off_delay: Optional[int] = None,
        fixed_b=True,
        fixed_phase=True,
        physical_qubit=None,
        file_path_prefix="",
        b_add_z_measurement=False,
        parameters_suffix="",
        b_kappa=False,
    ):
        """
        Args:

        physical_qubit: The qubit on which to run the Parity Oscillations experiment.
        osc_freq: The oscillation frequency induced by the user through a virtual
            Rz rotation. This quantity is given in Hz.
        cut_off_delay: The cut-off index of the delay. This argument limits the delays for the fourier analysis.
        fixed_b: fix equal probability of even/odd parity in the fitter
        fixed_phase: fix initial phase of zero in the fitter
        file_path_prefix: path to save the data
        b_add_z_measurement: measure along the z axis, in addition to the ramseyXY experiment
        parameters_suffix: string which added in the end of analysis results parameters
        b_kappa: Adding Gaussian decay envelope in the fitter

        """

        if not b_add_z_measurement:
            if (
                not b_kappa
            ):  # standard fitting model, two frequencies with exponential decay
                super().__init__(
                    models=[
                        lmfit.models.ExpressionModel(
                            expr="(amp * exp(-x / t2) * ( b * cos(2 * pi * freq_e * x + phase) + (1-b) * cos(2 * pi * freq_o * x + phase) ) + base)",
                            name="X",
                            data_sort_key={"series": "X"},
                        ),
                        lmfit.models.ExpressionModel(
                            expr="(amp * exp(-x / t2) * (b * sin(2 * pi * freq_e * x + phase) + (1-b) * sin(2 * pi * freq_o * x + phase)) + base)",
                            name="Y",
                            data_sort_key={"series": "Y"},
                        ),
                    ]
                )
            else:  # fit model with Gaussian envelope
                super().__init__(
                    models=[
                        lmfit.models.ExpressionModel(
                            expr="(amp * exp(-x / t2) * exp(-(x * kappa)**2) * ( b * cos(2 * pi * freq_e * x + phase) + (1-b) * cos(2 * pi * freq_o * x + phase) ) + base)",
                            name="X",
                            data_sort_key={"series": "X"},
                        ),
                        lmfit.models.ExpressionModel(
                            expr="(amp * exp(-x / t2) * exp(-(x * kappa)**2) * (b * sin(2 * pi * freq_e * x + phase) + (1-b) * sin(2 * pi * freq_o * x + phase)) + base)",
                            name="Y",
                            data_sort_key={"series": "Y"},
                        ),
                    ]
                )
        else:  # add z axis
            super().__init__(
                models=[
                    lmfit.models.ExpressionModel(
                        expr="(amp * exp(-x / t2) * ( b * cos(2 * pi * freq_e * x + phase) + (1-b) * cos(2 * pi * freq_o * x + phase) ) + base)",
                        name="X",
                        data_sort_key={"series": "X"},
                    ),
                    lmfit.models.ExpressionModel(
                        expr="(amp * exp(-x / t2) * (b * sin(2 * pi * freq_e * x + phase) + (1-b) * sin(2 * pi * freq_o * x + phase)) + base)",
                        name="Y",
                        data_sort_key={"series": "Y"},
                    ),
                    lmfit.models.ExpressionModel(
                        expr="(az * exp(-x / t1) + bz)",
                        name="Z",
                        data_sort_key={"series": "Z"},
                    ),
                ]
            )

        self._osc_freq = osc_freq
        self._cut_off_delay = cut_off_delay
        self._options.result_parameters = [
            curve.ParameterRepr("freq_e", "freq_e", "Hz"),
            curve.ParameterRepr("freq_o", "freq_o", "Hz"),
            curve.ParameterRepr("b", "b_PO"),
            curve.ParameterRepr("t2", "T2_PO", "S"),
            curve.ParameterRepr("amp", "A_PO"),
            curve.ParameterRepr("base", "B_PO"),
            curve.ParameterRepr("phase", "phi_PO"),
        ]
        if b_kappa:
            self._options.result_parameters.append(curve.ParameterRepr("kappa"))
        self._options.data_processor = DataProcessor(
            input_key="counts", data_actions=[Probability(outcome="0")]
        )

        self.fixed_b = fixed_b
        self.fixed_phase = fixed_phase
        self._options.fixed_parameters = dict()
        self.b_kappa = b_kappa

        self._plot_fourier = False  # if set True, plots the FFT of the signal

        self.physical_qubit = physical_qubit
        self._file_path_prefix = file_path_prefix
        self._save_data = True if self._file_path_prefix != "" else False
        self.b_add_z_measurement = b_add_z_measurement

        self.parameters_suffix = parameters_suffix

    def _run_analysis(self, experiment_data):
        saved_data = {}
        # check and set fix parameters for the fitter
        if self.fixed_b:
            self._options.fixed_parameters.update({"b": 0.5})
        if self.fixed_phase:
            self._options.fixed_parameters.update({"phase": 0})

        # Prepare for fitting
        self._initialize(experiment_data)
        analysis_results = []

        # Run data processing
        processed_data = self._run_data_processing(
            raw_data=experiment_data.data(),
            models=self._models,
        )

        if self.options.plot and self.options.plot_raw_data:
            for model in self._models:
                sub_data = processed_data.get_subset_of(model._name)
                self.plotter.set_series_data(
                    series_name=model._name,
                    x=sub_data.x,
                    y=sub_data.y,
                )

        # Format data
        formatted_data = self._format_data(processed_data)
        if self.options.plot:
            for model in self._models:
                sub_data = formatted_data.get_subset_of(model._name)
                saved_data[model._name] = sub_data
                self.plotter.set_series_data(
                    series_name=model._name,
                    x_formatted=sub_data.x,
                    y_formatted=sub_data.y,
                    y_formatted_err=sub_data.y_err,
                )

        # Run fitting
        fit_data = self._run_curve_fit(
            curve_data=formatted_data,
            models=self._models[:2],  # doesn't fit z axis
        )

        if fit_data.success:
            quality = self._evaluate_quality(fit_data)
            self.plotter.set_supplementary_data(fit_red_chi=fit_data.reduced_chisq)
        else:
            quality = "bad"

        if self.options.return_fit_parameters:
            # Store fit status overview entry regardless of success.
            # This is sometime useful when debugging the fitting code.
            overview = AnalysisResultData(
                name=PARAMS_ENTRY_PREFIX + self.name,
                value=fit_data,
                quality=quality,
                extra=self.options.extra,
            )
            analysis_results.append(overview)

        # Create analysis results
        analysis_results.extend(
            self._create_analysis_results(
                fit_data=fit_data, quality=quality, **self.options.extra.copy()
            )
        )

        # Draw fit curves and report
        interp_x = np.linspace(
            np.min(formatted_data.x), np.max(formatted_data.x), num=300
        )
        saved_data["time-fit"] = interp_x
        if self.options.plot:
            for model in self._models[:2]:
                y_data_with_uncertainty = eval_with_uncertainties(
                    x=interp_x,
                    model=model,
                    params=fit_data.ufloat_params,
                )
                y_mean = unp.nominal_values(y_data_with_uncertainty)
                saved_data[model._name + "fit"] = y_mean
                # Draw fit line
                self.plotter.set_series_data(
                    series_name=model._name,
                    x_interp=interp_x,
                    y_interp=y_mean,
                )
                if fit_data.covar is not None:
                    # Draw confidence intervals with different n_sigma
                    sigmas = unp.std_devs(y_data_with_uncertainty)
                    if np.isfinite(sigmas).all():
                        self.plotter.set_series_data(
                            series_name=model._name,
                            y_interp_err=sigmas,
                        )

        # Add raw data points
        if self.options.return_data_points:
            analysis_results.extend(
                self._create_curve_data(curve_data=formatted_data, models=self._models)
            )

        figures = []

        # use the fitted frequencies to calculated nu and delta
        freq_e = next(filter(lambda res: res.name == "freq_e", analysis_results)).value
        freq_o = next(filter(lambda res: res.name == "freq_o", analysis_results)).value
        delta = (freq_e + freq_o) / 2 - self._osc_freq
        nu = (freq_e - freq_o) / 2
        if unp.nominal_values(nu) < 0:
            nu = nu * (-1)
            for res in analysis_results:
                if res.name == "b_PO":
                    res.value = 1 - res.value

        parity_result = [
            AnalysisResultData(
                name="Delta_PO",
                value=delta,
                chisq=fit_data.reduced_chisq,
                quality=None,
                extra={"unit": "Hz"},
            ),
            AnalysisResultData(
                name="nu_PO",
                value=nu,
                chisq=fit_data.reduced_chisq,
                quality=None,
                extra={"unit": "Hz"},
            ),
        ]
        # Return combined results
        analysis_results = parity_result + analysis_results

        # Write fitting report
        names_to_report = ["Delta_PO", "nu_PO", "b_PO", "T2_PO", "phi_PO"]
        if self.b_kappa:
            names_to_report.append("kappa")
        if self.fixed_b:
            names_to_report.remove("b_PO")
        if self.fixed_phase:
            names_to_report.remove("phi_PO")
        report_results = []
        for res in analysis_results:
            if res.name in names_to_report:
                report_results.append(res)
        self.plotter.set_supplementary_data(primary_results=report_results)

        if self.options.plot:
            figures.append(self.plotter.figure())
            figures[0].axes[0].set_title(
                f"Driven freq = {round(self._osc_freq / 1e3, 1)}kHz"
            )

            if self._plot_fourier:
                figures.append(self._fourier_fig(experiment_data=experiment_data))

        if self._save_data:  # save data locally
            import pickle

            with open(
                self._file_path_prefix
                + ".Qubit_"
                + str(self.physical_qubit)
                + ".parity_data.pkl",
                "wb",
            ) as handle:
                pickle.dump(saved_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.parameters_suffix:  # add suffix if relevant
            new_analysis_results = []
            for an in analysis_results:
                new_analysis_results.append(
                    AnalysisResultData(
                        name=an.name + self.parameters_suffix,
                        value=an.value,
                        chisq=an.chisq,
                        quality=an.quality,
                        extra=an.extra,
                    )
                )
            analysis_results = new_analysis_results

        return analysis_results, figures

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.plotter.set_figure_options(
            xlabel="Delay",
            ylabel="P(0)",
            xval_unit="s",
        )
        default_options.result_parameters = []

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

        # find with FFT the 2 most dominant frequencies and set them as a guess to the fitter
        guesses_dict = dict()
        for series in ["X", "Y"]:
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

        amp_guess = guesses_dict["X"]["zero_delay"] - guesses_dict["Y"]["zero_delay"]
        base_guess = guesses_dict["Y"]["zero_delay"]
        freq_e_guess = (
            guesses_dict["X"]["top_freqs"][0] + guesses_dict["Y"]["top_freqs"][0]
        ) / 2
        freq_o_guess = (
            guesses_dict["X"]["top_freqs"][1] + guesses_dict["Y"]["top_freqs"][1]
        ) / 2

        user_opt.p0.set_if_empty(
            b=0.5,
            amp=amp_guess,
            base=base_guess,
        )
        # bound to the model
        user_opt.bounds.set_if_empty(
            amp=(0.3, 0.55),
            b=(0.3, 0.7),
            base=(0.3, 0.6),
            phase=(-0.1, 0.1),
        )

        # guess all frequencies signs options
        # other parameters can be modified or remove as needed
        options = []
        for i in (-1, 1):
            for j in (-1, 1):
                for t2_guess in (50e-6, 100e-6, 200e-6):
                    for phase_guess in (-0.03, 0, 0.03):
                        opt = user_opt.copy()
                        opt.p0.set_if_empty(
                            freq_e=i * freq_e_guess,
                            freq_o=j * freq_o_guess,
                            t2=t2_guess,
                            phase=phase_guess,
                        )
                        opt.bounds.set_if_empty(t2=(t2_guess / 5, t2_guess * 2))
                        if self.b_kappa:
                            opt.p0.set_if_empty(kappa=0)
                            opt.bounds.set_if_empty(kappa=(0, 1e4))
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
            curve.utils.is_error_not_significant(fit_freq),
        ]

        if all(criteria):
            return "good"

        return "bad"

    def _fourier_fig(self, experiment_data):
        # Run data processing
        processed_data = self._run_data_processing(
            raw_data=experiment_data.data(),
            models=self._models,
        )

        # Format data
        formatted_data = self._format_data(processed_data)
        data_RamX = formatted_data.get_subset_of("X")
        data_RamY = formatted_data.get_subset_of("Y")

        if self._cut_off_delay is not None:
            fourier_signal_x, freq_vec_x = myfft(
                data_RamX.y[: self._cut_off_delay]
                - np.mean(data_RamX.y[: self._cut_off_delay]),
                data_RamX.x[1] - data_RamX.x[0],
            )
            fourier_signal_y, freq_vec_y = myfft(
                data_RamY.y[: self._cut_off_delay]
                - np.mean(data_RamY.y[: self._cut_off_delay]),
                data_RamY.x[1] - data_RamY.x[0],
            )
        else:
            fourier_signal_x, freq_vec_x = myfft(
                data_RamX.y - np.mean(data_RamX.y), data_RamX.x[1] - data_RamX.x[0]
            )
            fourier_signal_y, freq_vec_y = myfft(
                data_RamY.y - np.mean(data_RamY.y), data_RamY.x[1] - data_RamY.x[0]
            )
        fig_fourier = Figure()
        _ = FigureCanvasSVG(fig_fourier)
        ax = fig_fourier.subplots(1, 1)
        ax.plot(freq_vec_x / 1e3, abs(fourier_signal_x), label="X")
        ax.plot(freq_vec_y / 1e3, abs(fourier_signal_y), label="Y")
        ax.set_xlim([0, max(freq_vec_x / 1e3)])
        ax.set_xlabel("frequency [kHz]")
        ax.set_ylabel("abs(FFT)")
        ax.legend()

        return fig_fourier
