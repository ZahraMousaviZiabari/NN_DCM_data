from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

def print_regression_results(coefs_time, coefs_wait, x_names = [""]):
    names = ["Intercept"] + x_names

    print("\n=== Regression on pred_vots ===")
    for name, coef in zip(names, coefs_time):
        print(f"{name:15s} : {coef: .6f}")

    print("\n=== Regression on pred_vohts ===")
    for name, coef in zip(names, coefs_wait):
        print(f"{name:15s} : {coef: .6f}")
        
def regress(pred_vots, pred_vowts, df, base_vars):
    # Z a 2D array N,K
    # Base predictors
    Z_parts = [df[var].to_numpy().reshape(-1, 1) for var in base_vars]
    
    # Final design matrix
    Z = np.hstack(Z_parts)

    # Fit regressions
    reg_vot  = LinearRegression().fit(Z, pred_vots)
    reg_vowt = LinearRegression().fit(Z, pred_vowts)

    # Collect coefficients (intercept first)
    coefs_time = [reg_vot.intercept_] + list(reg_vot.coef_)
    coefs_wait = [reg_vowt.intercept_] + list(reg_vowt.coef_)

    return coefs_time, coefs_wait


def value_of_x(df, coef_time, coef_wait, base_vars, interactions=None):
    """
    df            : pandas DataFrame with predictor columns
    coef_time     : array of coefficients (intercept + slopes)
    coef_wait     : array of coefficients (intercept + slopes)
    base_vars     : list of base column names (strings)
    interactions  : list of tuples specifying interaction columns
                    Example: [('INC','FULL'), ('INC','FLEX')]
    """

    # Base predictors
    Z_parts = [df[var].to_numpy().reshape(-1, 1) for var in base_vars]

    # Interaction predictors
    if interactions:
        for i, j in interactions:
            var1 = base_vars[i]
            var2 = base_vars[j]
            Z_parts.append(
                (df[var1] * df[var2]).to_numpy().reshape(-1, 1)
            )

    # Final design matrix
    Z = np.hstack(Z_parts)

    # Compute outcomes (vectorized)
    vots  = Z @ coef_time[1:] + coef_time[0]
    vowts = Z @ coef_wait[1:] + coef_wait[0]

    return vots, vowts

def plotVOT(pred_vots, x_values, legend_arr, result_path, simulate = False):
    
    fg, ax = plt.subplots()
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]
    pred_sorted = pred_vots[sort_idx]
    ax.plot(x_sorted, pred_sorted)

    ax.set_title("Predicted")
    legend_labels = [', '.join(row) for row in legend_arr]
    ax.legend(legend_labels)
    ax.set_xlabel("income (Franc per min)", fontsize=12)
    ax.set_ylabel("value of time (Franc per min)", fontsize=12)

    os.makedirs(result_path, exist_ok=True)
    fg.savefig(result_path + "/" + "VOT_vs_inc.png", dpi=250)
    plt.close()


def plotVOHE(pred_vohes, x_values, legend_arr, result_path, simulate = False):
    
    fg, ax = plt.subplots()
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]
    pred_sorted = pred_vohes[sort_idx]
    ax.plot(x_sorted, pred_sorted)


    ax.set_title("Predicted")
    legend_labels = [', '.join(row) for row in legend_arr]
    ax.legend(legend_labels)
    ax.set_xlabel("income (Franc per min)", fontsize=12)
    ax.set_ylabel("value of headway (Frac per min)", fontsize=12)

    os.makedirs(result_path, exist_ok=True)
    fg.savefig(result_path + "/" + "VOHE_vs_inc.png", dpi=250)
    plt.close()
    

def boxplot_VOT(X_masked, pred_vots, result_path):
    fg, ax = plt.subplots()
    grouped = [
    X_masked.loc[X_masked['INCOME_GROUP'] == g, pred_vots]
    for g in ['0', '1', '2']
    ]

    ax.boxplot(grouped, labels=['INCOME_0', 'INCOME_1', 'INCOME_2'])
    ax.set_ylabel('Predicted VOT')
    ax.set_title(f'Boxplot of VoTs by Income Group')
    os.makedirs(result_path, exist_ok=True)
    fg.savefig(result_path + "/" + "boxplot_VOT_vs_inc.png", dpi=250)
    
def boxplot_VOHE(X_masked, pred_vohs, result_path):
    fg, ax = plt.subplots()
    grouped = [
    X_masked.loc[X_masked['INCOME_GROUP'] == g, pred_vohs]
    for g in ['0', '1', '2']
    ]

    ax.boxplot(grouped, labels=['INCOME_0', 'INCOME_1', 'INCOME_2'])
    ax.set_ylabel('Predicted VOHE')
    ax.set_title(f'Boxplot of VOHEs by Income Group')
    os.makedirs(result_path, exist_ok=True)
    fg.savefig(result_path + "/" + "boxplot_VOHE_vs_inc.png", dpi=250)
    
       

if __name__ == "__main__":
    # === Load data ===
    with open("results_one_run.pkl", "rb") as f:
        results_summary = pickle.load(f)

    # Extract relevant data
    mean_estimated_betas_df = results_summary["mean_estimated_betas"]
    std_estimated_betas_df = results_summary["std_estimated_betas"]
    final_estimated_asc = results_summary["estimated_asc"]
    f1_macro = results_summary["f1_macro"]
    accuracy = results_summary["accuracy"]
    ll_train = results_summary["ll_train"]
    ll_test = results_summary["ll_test"]

    print("\n----Outputs and Performance Metrics----")
    # Compute performance metrics
    print(f"Average Log Likelihood Loss (Train): {ll_train:.4f}")
    print(f"Average Log Likelihood Loss (Test): {ll_test:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Mean Estimated Betas:\n", mean_estimated_betas_df)
    print("Std Estimated Betas:\n", std_estimated_betas_df)
    print("Final Estimated ASCs:\n", final_estimated_asc)
    print(f"Prediction F1-score (macro): {f1_macro:.2f}%")
    
