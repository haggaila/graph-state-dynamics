# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import os
import pickle
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp


# unumpy shortcuts
unp_n = unp.nominal_values
unp_s = unp.std_devs

# Set plot parameters
plt.rc("text", usetex=True)  # set usetex=False if tex isn't installed
plt.rc("font", family="serif")
plt.rcParams["font.size"] = 16
plt.rcParams["axes.linewidth"] = 2

s_output_path = (
    os.path.abspath("../output") + "/"
)  # use the absolute path of the current file
s_PO_path = s_output_path + "PO/"

# First figure
qubit = 2
file = (
    s_PO_path
    + f"ibm_cusco.nu.89337de1544b4ba080bea1cc50c59391.Qubit_{qubit}.parity_data.pkl"
)
# Load data
with open(file, "rb") as handle:
    data_1Q = pickle.load(handle)

fig, axs = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
ax = axs[0]
# calculate the xy projection, rho
p_x = unp.uarray(data_1Q["X"].y, data_1Q["X"].y_err)
p_y = unp.uarray(data_1Q["Y"].y, data_1Q["Y"].y_err)
rho = unp.sqrt((2 * p_x - 1) ** 2 + (2 * p_y - 1) ** 2)
# calculate <Z>
p_z = unp.uarray(data_1Q["Z"].y, data_1Q["Z"].y_err)
z_ev = 2 * p_z - 1
# plot
ax.errorbar(
    data_1Q["X"].x * 1e6,
    unp_n(z_ev),
    yerr=unp_s(z_ev),
    fmt=">",
    alpha=0.8,
    capsize=4,
    markersize=4,
    label=r"$\left< Z \right>,$ exp ",
    color="blue",
)
ax.errorbar(
    data_1Q["X"].x * 1e6,
    unp_n(rho),
    yerr=unp_s(rho),
    fmt="o",
    alpha=0.9,
    capsize=4,
    markersize=4,
    label=r"$\sqrt{\left< X \right>^2 + \left< Y \right>^2},$ exp ",
    color="red",
)
ax.xaxis.set_tick_params(which="major", size=7, width=2, direction="in", top="on")
ax.xaxis.set_tick_params(which="minor", size=5, width=2, direction="in", top="on")
ax.yaxis.set_tick_params(which="major", size=7, width=2, direction="in", right="on")
ax.yaxis.set_tick_params(which="minor", size=5, width=2, direction="in", right="on")
ax.set_ylim([-0.11, 1.1])
ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=14)
ax = axs[1]

ax.errorbar(
    data_1Q["X"].x * 1e6,
    data_1Q["X"].y,
    yerr=data_1Q["X"].y_err,
    fmt="o",
    alpha=0.9,
    capsize=4,
    markersize=5,
    label=r"$P_x,$ exp",
    color="C0",
)
ax.errorbar(
    data_1Q["time-fit"] * 1e6,
    data_1Q["Xfit"],
    fmt="--",
    alpha=0.8,
    capsize=4,
    markersize=5,
    label=r"$P_x,$ fit",
    color="C0",
)

ax.errorbar(
    data_1Q["Y"].x * 1e6,
    data_1Q["Y"].y,
    yerr=data_1Q["Y"].y_err,
    fmt=">",
    alpha=0.9,
    capsize=4,
    markersize=5,
    label=r"$P_y,$ exp",
    color="C1",
)
ax.errorbar(
    data_1Q["time-fit"] * 1e6,
    data_1Q["Yfit"],
    fmt="-b",
    alpha=0.8,
    capsize=4,
    markersize=5,
    label=r"$P_y,$ fit",
    color="C1",
)
# Edit the major and minor ticks of the x and y axes
ax.xaxis.set_tick_params(which="major", size=7, width=2, direction="in", top="on")
ax.xaxis.set_tick_params(which="minor", size=5, width=2, direction="in", top="on")
ax.yaxis.set_tick_params(which="major", size=7, width=2, direction="in", right="on")
ax.yaxis.set_tick_params(which="minor", size=5, width=2, direction="in", right="on")
ax.set_ylim([0.08, 1.15])
ax.set_xlabel(r"time [$\mu s$]", labelpad=1, fontsize=18)
ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=14)
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.show()

# Second figure
qubit = 45
file = (
    s_PO_path
    + f"ibm_cusco.nu.8409cd5723dd430993e499d402e029bb.Qubit_{qubit}.parity_data.pkl"
)
# Load data
with open(file, "rb") as handle:
    data_1Q = pickle.load(handle)

fig, axs = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
ax = axs
p_x = unp.uarray(data_1Q["X"].y, data_1Q["X"].y_err)
p_y = unp.uarray(data_1Q["Y"].y, data_1Q["Y"].y_err)
rho = unp.sqrt((2 * p_x - 1) ** 2 + (2 * p_y - 1) ** 2)

ax.errorbar(
    data_1Q["X"].x * 1e6,
    data_1Q["X"].y,
    yerr=data_1Q["X"].y_err,
    fmt="o",
    alpha=0.9,
    capsize=4,
    markersize=5,
    label=r"$P_x,$ exp",
    color="C0",
)
ax.errorbar(
    data_1Q["time-fit"] * 1e6,
    data_1Q["Xfit"],
    fmt="--",
    alpha=0.8,
    capsize=4,
    markersize=5,
    label=r"$P_x,$ fit",
    color="C0",
)

ax.errorbar(
    data_1Q["Y"].x * 1e6,
    data_1Q["Y"].y,
    yerr=data_1Q["Y"].y_err,
    fmt=">",
    alpha=0.9,
    capsize=4,
    markersize=5,
    label=r"$P_y,$ exp",
    color="C1",
)
ax.errorbar(
    data_1Q["time-fit"] * 1e6,
    data_1Q["Yfit"],
    fmt="-b",
    alpha=0.8,
    capsize=4,
    markersize=5,
    label=r"$P_y,$ fit",
    color="C1",
)
# Edit the major and minor ticks of the x and y axes
ax.xaxis.set_tick_params(which="major", size=7, width=2, direction="in", top="on")
ax.xaxis.set_tick_params(which="minor", size=5, width=2, direction="in", top="on")
ax.yaxis.set_tick_params(which="major", size=7, width=2, direction="in", right="on")
ax.yaxis.set_tick_params(which="minor", size=5, width=2, direction="in", right="on")
ax.set_ylim([0.0, 1.05])
ax.set_xlabel(r"time [$\mu s$]", labelpad=1, fontsize=18)
ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=14)
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.show()
