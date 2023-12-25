import os
import matplotlib.pyplot as plt
from project_experiments.load_routines import query_db

# Set plot parameters
plt.rc("text", usetex=True)  # set usetex=False if tex isn't installed
plt.rc("font", family="serif")
plt.rcParams["font.size"] = 16
plt.rcParams["axes.linewidth"] = 2

# Files with the fit results
s_db_filename_1 = "ibm_cusco.nu.dde4848bb5cb4792803098f681799afddata.csv"
s_db_filename_2 = "ibm_cusco.nu.3db06923872a4c8eaa9f20dfe164579ddata.csv"

s_db_filename_1_b = "ibm_cusco.nu.8e4f75fb6fd44caf95bcc57c96e54994data.csv"
s_db_filename_2_b = "ibm_cusco.nu.73bac065c0fa4acebb7b0ea55b0f4b46data.csv"

s_db_filename_1_kappa = "ibm_cusco.nu.9de0c3cedcb04dd1967c7a873437b9addata.csv"
s_db_filename_2_kappa = "ibm_cusco.nu.f62de71eb69346139fac3086ce4d5f62data.csv"

s_output_path = os.path.abspath("../output") + "/"
s_csv_file = s_output_path + "PO/" + s_db_filename_1

# Arbitrary fit parameter, just for load the reduced-chi^2
parameter = "'nu_PO'"

sort_by = ["end_date"]

# Load files
s_filter_query = "variable_name == " + parameter
df1 = query_db(s_csv_file, s_filter_query, sort_by, na_position="first")

s_csv_file = s_output_path + "PO/" + s_db_filename_2
df2 = query_db(s_csv_file, s_filter_query, sort_by, na_position="first")

s_csv_file = s_output_path + "PO/" + s_db_filename_1_b
df1_b = query_db(s_csv_file, s_filter_query, sort_by, na_position="first")

s_csv_file = s_output_path + "PO/" + s_db_filename_2_b
df2_b = query_db(s_csv_file, s_filter_query, sort_by, na_position="first")

s_csv_file = s_output_path + "PO/" + s_db_filename_1_kappa
df1_kappa = query_db(s_csv_file, s_filter_query, sort_by, na_position="first")

s_csv_file = s_output_path + "PO/" + s_db_filename_2_kappa
df2_kappa = query_db(s_csv_file, s_filter_query, sort_by, na_position="first")

# Plot reduce chi^2
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(df1_kappa.chi_sq, df1.chi_sq, color="blue", alpha=0.5)
ax.scatter(df2_kappa.chi_sq, df2.chi_sq, color="blue", alpha=0.5)
ax.set_xlabel(r"Reduced $\chi^2$ with fitting $\kappa$", labelpad=1, fontsize=18)
ax.set_ylabel(r"Reduced $\chi^2$ with fixing $\kappa=0$", labelpad=1, fontsize=18)
ax.xaxis.set_tick_params(which="major", size=7, width=2, direction="in", top="on")
ax.xaxis.set_tick_params(which="minor", size=5, width=2, direction="in", top="on")
ax.yaxis.set_tick_params(which="major", size=7, width=2, direction="in", right="on")
ax.yaxis.set_tick_params(which="minor", size=5, width=2, direction="in", right="on")
ax.set_ylim([0, 10])
ax.set_xlim([0, 10])
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(df1_b.chi_sq, df1.chi_sq, color="blue", alpha=0.5)
ax.scatter(df2_b.chi_sq, df2.chi_sq, color="blue", alpha=0.5)
ax.set_xlabel(r"Reduced $\chi^2$ with fitting $b$", labelpad=1, fontsize=18)
ax.set_ylabel(r"Reduced $\chi^2$ with fixing $b=1/2$", labelpad=1, fontsize=18)
ax.xaxis.set_tick_params(which="major", size=7, width=2, direction="in", top="on")
ax.xaxis.set_tick_params(which="minor", size=5, width=2, direction="in", top="on")
ax.yaxis.set_tick_params(which="major", size=7, width=2, direction="in", right="on")
ax.yaxis.set_tick_params(which="minor", size=5, width=2, direction="in", right="on")
ax.set_ylim([0, 10])
ax.set_xlim([0, 10])
plt.tight_layout()
plt.show()
