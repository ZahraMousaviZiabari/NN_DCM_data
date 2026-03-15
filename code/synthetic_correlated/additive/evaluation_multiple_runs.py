import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

vots_list = []
vowts_list = []
#== Load data from multiple runs ===
dir = "multiple_runs"
with open(dir + "/" +'vot_vowt_mean_std_runs.pkl', 'rb') as f:
    vot_vowt_runs = pickle.load(f)
with open(dir + "/" +'results_multiple_runs.pkl', 'rb') as f:
    results_multiple_runs = pickle.load(f)

#=== Extract summary data ===
summary_performance = results_multiple_runs["summary_performance"]
summary_parameter_estimation = results_multiple_runs["summary_parameter_estimation"]
summary_coefs_errors = results_multiple_runs["summary_coefs_errors"]
summary_vots_errors = results_multiple_runs["summary_vots_errors"]
summary_vowts_errors = results_multiple_runs["summary_vowts_errors"]

# === Print summary data ===
print("\n=== Summary of performance metrics ===")
for key, value in summary_performance.items():
    print(f"{key}: {value:.6f}")

print("\n=== Summary of parameter estimation ===")
names = [
        "Intercept",
        "INC",
        "FULL",
        "FLEX",
        "INC × FULL",
        "INC × FLEX",
        "FULL × FLEX"
    ]
for key, values in summary_parameter_estimation.items():
    print(f"{key}")
    for name, value in zip(names, values):
        print(f"{name:15s} : {value: .6f}")

print("\n=== Summary of error of coefs ===")
for key, values in summary_coefs_errors.items():
    print(f"{key}")
    for k, v in values.items():
        print(f"{k}: {v:.6f}")

print("\n=== Summary of error of vots ===")
for key, values in summary_vots_errors.items():
    print(f"{key}")
    for k, v in values.items():
        print(f"{k}: {v:.6f}")

print("\n=== Summary of error of vowts ===")
for key, values in summary_vowts_errors.items():
    print(f"{key}")
    for k, v in values.items():
        print(f"{k}: {v:.6f}")


# === VoT VoWT Visualization - Simulation (deterministic) data ===
mean_vots = vot_vowt_runs["mean_vots"]
std_vots  = vot_vowt_runs["std_vots"] 
mean_vowts = vot_vowt_runs["mean_vowts"]
std_vowts  = vot_vowt_runs["std_vowts"]

N = len(mean_vots[0])
inc_vals = np.linspace(0, 1.0, N).reshape(-1,1)

inc = inc_vals *60
# === Save setup ===
save_dir = "results_plots/multiple_runs"
os.makedirs(save_dir, exist_ok=True)
base_var_sets = [
        ['FULL', 'FLEX'],
        ['FULL', 'NOFLEX'],
        ['NOFULL', 'FLEX'],
        ['NOFULL', 'NOFLEX']
    ]
plt.figure(figsize=(6 ,6))

legend_handles = []

for i in range(4):
    handle = Line2D(
        [0], [0],
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        linewidth=2,
        marker='|',          # vertical bar
        markersize=15,
        markeredgewidth=1,
        alpha=0.7
    )
    legend_handles.append(handle)

for i in range(len(mean_vots)):
    line, = plt.plot(inc, mean_vots[i])
    color = line.get_color()
    plt.errorbar(
    inc,
    mean_vots[i],
    yerr=std_vots[i],
    fmt="none",
    ecolor=color,
    elinewidth=1.5,
    alpha=0.7
    )

legend_labels = [', '.join(row) for row in base_var_sets]
plt.xlabel("income ($ per hour)")
plt.ylabel("value of time ($ per hour)")
plt.legend(
    handles=legend_handles,
    labels=legend_labels
)
plt.title("Additive (mean, std)")
plt.tight_layout()
plt.savefig(save_dir + "/" +'vots_all_runs.png', dpi=300)
plt.close()


plt.figure(figsize=(6 ,6))

for i in range(len(mean_vowts)):
    line, = plt.plot(inc, mean_vowts[i])
    color = line.get_color()
    plt.errorbar(
    inc,
    mean_vowts[i],
    yerr=std_vowts[i],
    fmt="none",
    ecolor=color,
    elinewidth=1.5,
    alpha=0.7
    )

plt.xlabel("income ($ per hour)")
plt.ylabel("value of wait ($ per hour)")
plt.legend(
    handles=legend_handles,
    labels=legend_labels
)
plt.title("Additive (mean, std)")
plt.tight_layout()
plt.savefig(save_dir + "/" + 'vowts_all_runs.png', dpi=300)
plt.close()


with open('summary_cv5.pkl', 'rb') as f:
    vot_vowt_cv5 = pickle.load(f)


# === VoT VoWT cross validation statistics===
cv_vot = vot_vowt_cv5["vot"]
cv_vowt  = vot_vowt_cv5["vowt"] 
cv_accuracy = vot_vowt_cv5["accuracy"]
cv_ll_train  = vot_vowt_cv5["ll_train"]
cv_ll_test = vot_vowt_cv5["ll_test"]

# === Convert to arrays ===
vots_arr = -np.stack(cv_vot, axis=0)# shape (splits, r, c)
vowts_arr = -np.stack(cv_vowt, axis=0)


mean_vots_arr = vots_arr.mean(axis=0)  # shape: (runs, n_x)
var_vots_arr  = vots_arr.var(axis=0)   
mean_vowts_arr = vowts_arr.mean(axis=0) 
var_vowts_arr  = vowts_arr.var(axis=0)  


print("\n--Averages over cross validation runs:")
print(f"Mean log-likelihood (train): {np.mean(cv_ll_train):.6f}")
print(f"Mean log-likelihood (test): {np.mean(cv_ll_test):.6f}")
print(f"Mean E[vot]: {np.mean(mean_vots_arr):.6f}")
print(f"Mean E[vowt]: {np.mean(mean_vowts_arr):.6f}")
print(f"Mean var[vot]: {np.mean(var_vots_arr):.6f}")
print(f"Mean var[vowt]: {np.mean(var_vowts_arr):.6f}")
print(f"Mean accuracy: {np.mean(cv_accuracy):.6f}")

print("--Std over cross validation runs:")
print(f"std log-likelihood (train): {np.std(cv_ll_train):.6f}")
print(f"std log-likelihood (test): {np.std(cv_ll_test):.6f}")
print(f"std E[vot]: {np.std(mean_vots_arr):.6f}")
print(f"std E[vowt]: {np.std(mean_vowts_arr):.6f}")
print(f"std var[vot]: {np.std(var_vots_arr):.6f}")
print(f"std var[vowt]: {np.std(var_vowts_arr):.6f}")
print(f"std accuracy: {np.std(cv_accuracy):.6f}")

