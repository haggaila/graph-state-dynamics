# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import pandas as pd
import matplotlib.pyplot as plt
from project_experiments.load_routines import query_db

# Set plot parameters
plt.rc("text", usetex=True)  # set usetex=False if tex isn't installed
plt.rc("font", family="serif")
plt.rcParams["font.size"] = 18
plt.rcParams["axes.linewidth"] = 2

s_db_filename = "ibm_cusco.Time_Series.07_20_2023.9876f87948744fc0a060c645dcabc189.csv"
s_output_path = os.path.abspath("../output") + "/"
s_csv_file = s_output_path + "time_series/" + s_db_filename

# Choose which parameter to load
parameter = "'nu_PO'"

sort_by = ["end_date"]
s_group_by = "device_components"

# Choose how to filter the data
s_filter_query = (
    "variable_name == " + parameter + " & chi_sq < 3.0" + " & std_dev < 100000"
)
df = query_db(s_csv_file, s_filter_query, sort_by, na_position="first")

# Set specific qubits to plot
show_qubits = ["0", "10", "18", "20", "22", "24", "30", "8", "2", "12"]

# Calculate time of the experiment relative to the first one
dates = pd.to_datetime(df.end_date)
min_t = min(dates)
t_axis = []
for index, value in dates.items():
    t_axis.append((dates[index] - min_t).seconds / 60.0)
df["t"] = t_axis

groups = df.groupby(["device_components"])

# Plot
fig, axs = plt.subplots(
    3, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1, 1, 3]}, sharex=True
)
count = 0
for group_name, group_df in groups:
    if group_name[1:] in show_qubits:
        if group_name[1:] == "12":
            ax = axs[0]
        elif group_name[1:] == "2":
            ax = axs[1]
        else:
            ax = axs[2]
        ax.errorbar(
            group_df.t.values,
            group_df.value.values,
            yerr=group_df.std_dev.values,
            fmt="-o",
            alpha=0.8,
            capsize=4,
            markersize=5,
            label=str(group_name),
            color=f"C{count}",
        )
        count += 1

axs[-1].set_xlabel(r"time [m]")
for ax in axs:
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", frameon=False, fontsize=13)
    ax.xaxis.set_tick_params(which="major", size=7, width=2, direction="in", top="on")
    ax.xaxis.set_tick_params(which="minor", size=5, width=2, direction="in", top="on")
    ax.yaxis.set_tick_params(which="major", size=7, width=2, direction="in", right="on")
    ax.yaxis.set_tick_params(which="minor", size=5, width=2, direction="in", right="on")
fig.supylabel(r"$\nu / 2\pi$ [Hz]")
plt.tight_layout()
plt.subplots_adjust(hspace=0.08)
plt.show()
