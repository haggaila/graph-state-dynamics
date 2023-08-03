# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from typing import List

import numpy as np


class MCCube:
    """A cube used for drawing points and running a Monte Carlo integration."""

    def __init__(self, intervals: [(float, float)] = None, volume=0.0):
        """Constructor.

        Args:
                intervals: A list of tuples of two floats, defining the interval boundaries of
                    each parameter.
                volume: The total integration volume (separately calculated from the intervals).
        """
        self.intervals = intervals
        self.volume = volume
        self.values = []
        self.ordered_values = []
        self.ordered_indices = []
        self.zeros = None
        self.ones = None
        # self.probabilities = {}
        self.log_p = {}
        self.log_1_p = {}
        self.p_non_positive = {}
        self.p_non_ones = {}

    def draw(self, n_draws: int):
        """Draw points uniformly within the cube.

        Args:
                n_draws: The number of draws of points from the cube.
        """
        self.values = []
        for interval in self.intervals:
            values = np.random.uniform(interval[0], interval[1], n_draws)
            self.values.append(values)
        self.zeros = np.zeros(n_draws)
        self.ones = np.ones(n_draws)

    def delete_values(self, delete_indices: List):
        """Delete some indices from all cube members.

        Args:
                delete_indices: Indices to delete.
        """
        for i, values in enumerate(self.values):
            self.values[i] = np.delete(values, delete_indices)
        self.zeros = np.delete(self.zeros, delete_indices)
        self.ones = np.delete(self.ones, delete_indices)
        for i, _ in enumerate(self.ordered_indices):
            if self.ordered_indices[i] == -1:
                self.ordered_values[i] = self.zeros
            elif self.ordered_indices[i] == -2:
                self.ordered_values[i] = self.ones
            else:
                self.ordered_values[i] = self.values[self.ordered_indices[i]]
