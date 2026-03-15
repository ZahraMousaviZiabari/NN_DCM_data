import pickle
from simulate import error_of_vot, error_of_vowt, dic_z, inc
import copy
from data_utils import ChoiceDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

vots_list = []
vowts_list = []
dir = "results/data_CORR"
with open(dir + "/" + 'hundred_runs_hidden_60_l1_001_models.pkl', 'rb') as f:
    models = pickle.load(f)

# load data
data = pickle.load(open("toy_data/data_10k_rho_0.6.pkl", "rb"))
    
ds_train = ChoiceDataset(data['train'])
#ds_dev = ChoiceDataset(data['dev'])
ds_test = ChoiceDataset(data['test'])

input_z = copy.deepcopy(dic_z)

for model in models:
    sim_pred_vots, sim_true_vots, rmse, mabse, re = error_of_vot(model, dic_z, input_z, ds_train.params)
    sim_pred_vowts, sim_true_vowts, rmse, mabse, re = error_of_vowt(model, dic_z, input_z, ds_train.params)
    vots_list.append(np.array(sim_pred_vots, copy=True))
    vowts_list.append(np.array(sim_pred_vowts, copy=True))

vots_all_runs = np.stack(vots_list, axis=0)
vowts_all_runs = np.stack(vowts_list, axis=0)
print(vots_all_runs.shape)

# === VoT VoWT Visualization - Simulation (deterministic) data ===
mean_vots = vots_all_runs.mean(axis=0)  # shape: (runs, n_x)
std_vots  = vots_all_runs.std(axis=0)   
mean_vowts = vowts_all_runs.mean(axis=0) 
std_vowts  = vowts_all_runs.std(axis=0)   

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
inc = inc * 60

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
plt.title("TasteNet (mean, std)")
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
plt.title("TasteNet (mean, std)")
plt.tight_layout()
plt.savefig(save_dir + "/" + 'vowts_all_runs.png', dpi=300)
plt.close()


#=========== model 49
plt.figure(figsize=(6 ,6))
inc = inc * 60

for i in range(len(vots_all_runs[49])):
    line, = plt.plot(inc, vots_all_runs[49,i])



plt.xlabel("income ($ per hour)")
plt.ylabel("value of time ($ per hour)")
plt.legend(labels=legend_labels)
plt.title("TasteNet")
plt.tight_layout()
plt.savefig(save_dir + "/" +'vots_49.png', dpi=300)
plt.close()

plt.figure(figsize=(6 ,6))

for i in range(len(vots_all_runs[49])):
    line, = plt.plot(inc, vots_all_runs[49,i])

plt.xlabel("income ($ per hour)")
plt.ylabel("value of wait ($ per hour)")
plt.legend(labels=legend_labels)
plt.title("TasteNet")
plt.tight_layout()
plt.savefig(save_dir + "/" + 'vowts_49.png', dpi=300)
plt.close()

dir = "results"
with open(dir + "/" +'summary_cv5.pkl', 'rb') as f:
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


print("--Averages over runs:")
print(f"Mean E[vot]: {np.mean(mean_vots_arr):.6f}")
print(f"Mean E[vowt]: {np.mean(mean_vowts_arr):.6f}")
print(f"Mean var[vot]: {np.mean(var_vots_arr):.6f}")
print(f"Mean var[vowt]: {np.mean(var_vowts_arr):.6f}")
print(f"Mean accuracy: {np.mean(cv_accuracy):.6f}")

print("--Std over runs:")
print(f"std E[vot]: {np.std(mean_vots_arr):.6f}")
print(f"std E[vowt]: {np.std(mean_vowts_arr):.6f}")
print(f"std var[vot]: {np.std(var_vots_arr):.6f}")
print(f"std var[vowt]: {np.std(var_vowts_arr):.6f}")
print(f"std accuracy: {np.std(cv_accuracy):.6f}")