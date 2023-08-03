# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import List, Union
import lmfit
import numpy as np
from qiskit.providers.options import Options
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis import fit_function, CurveData


class ZZRamseyAnalysis(curve.CurveAnalysis):
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

    def __init__(self):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp * exp(-x / tau) * cos(2 * pi * (freq - zz) * x + phase + pi) + base",
                    name="0",
                    data_sort_key={"series": "0"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp * exp(-x / tau) * cos(2 * pi * (freq + zz) * x + phase + pi) + base",
                    name="1",
                    data_sort_key={"series": "1"},
                ),
            ]
        )

    @classmethod
    def _default_options(cls) -> Options:
        """Return the default analysis options.
        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """

        default_options = super()._default_options()
        default_options.plotter.set_figure_options(
            xlabel="Delay",
            ylabel="P(1)",
            xval_unit="s",
        )
        default_options.result_parameters = [curve.ParameterRepr("zz", "zz", "Hz")]

        return default_options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: CurveData,  # pylint: disable=unused-argument
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Compute the initial guesses.
        Args:
            user_opt: Fit options filled with user provided guess and bounds.
        Returns:
            List of fit options that are passed to the fitter function.
        """
        max_abs_y, _ = curve.guess.max_height(curve_data.y, absolute=True)

        user_opt.bounds.set_if_empty(
            amp=(-2 * max_abs_y, 2 * max_abs_y),
            tau=(0, np.inf),
            base=(-max_abs_y, max_abs_y),
            phase=(-np.pi, np.pi),
            freq=(0, np.inf),
        )

        # Guess the exponential decay by combining both curves
        data_0 = curve_data.get_subset_of("0")
        data_1 = curve_data.get_subset_of("1")

        # Default guess values
        freq_guesses = [
            curve.guess.frequency(data_0.x, data_0.y),
            curve.guess.frequency(data_1.x, data_1.y),
        ]
        base_guesses = [
            curve.guess.constant_sinusoidal_offset(data_0.y),
            curve.guess.constant_sinusoidal_offset(data_1.y),
        ]

        user_opt.p0.set_if_empty(
            tau=100e-6,
            amp=0.5,
            phase=0.0,
            freq=float(np.average(freq_guesses)),
            base=np.average(base_guesses),
            zz=(freq_guesses[1] - freq_guesses[0]) / 2,
        )

        return user_opt

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.
        A good fit has:
            - a reduced chi-squared lower than three,
            - an error on the frequency smaller than the frequency.
        """
        fit_freq = fit_data.fitval("freq")

        criteria = [
            fit_data.reduced_chisq < 3,
        ]

        if all(criteria):
            return "good"

        return "bad"
