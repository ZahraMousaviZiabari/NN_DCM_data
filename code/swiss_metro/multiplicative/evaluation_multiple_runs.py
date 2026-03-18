import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

vots_list = []
vowts_list = []
#== Load data from multiple runs ===
dir = "multiple_runs"
with open(dir + "/" +'results_multiple_runs.pkl', 'rb') as f:
    summary_performance = pickle.load(f)

# === Print summary data ===
print("\n=== Summary of performance metrics ===")
for key, value in summary_performance.items():
    print(f"{key}: {value:.6f}")

#=== Load cross validation data ===
with open('summary_cv5.pkl', 'rb') as f:
    summary_cv5 = pickle.load(f)


# === VoT VoWT cross validation statistics===
cv_f1  = summary_cv5["f1"] 
cv_accuracy = summary_cv5["accuracy"]
cv_ll_train  = summary_cv5["ll_train"]
cv_ll_test = summary_cv5["ll_test"]

print("\n--Averages over cross validation runs:")
print(f"Mean log-likelihood (train): {np.mean(cv_ll_train):.6f}")
print(f"Mean log-likelihood (test): {np.mean(cv_ll_test):.6f}")
print(f"Mean F1-score (macro): {np.mean(cv_f1):.6f}")
print(f"Mean accuracy: {np.mean(cv_accuracy):.6f}")

print("--Std over cross validation runs:")
print(f"std log-likelihood (train): {np.std(cv_ll_train):.6f}")
print(f"std log-likelihood (test): {np.std(cv_ll_test):.6f}")
print(f"std F1-score (macro): {np.std(cv_f1):.6f}")
print(f"std accuracy: {np.std(cv_accuracy):.6f}")

