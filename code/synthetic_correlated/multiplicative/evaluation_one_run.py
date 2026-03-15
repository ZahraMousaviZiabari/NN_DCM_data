import pickle
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def synthetic_sociodemo(N, p1):
    """
    Generate socioeconomic characteristics without correlations:
    """
    if p1 == 'deterministic':
        inc_vals = np.linspace(0, 1.0, N).reshape(-1,1)
        
        fulltime = np.ones((N, 1))
        nofulltime = np.zeros((N, 1))
        flex = np.ones((N, 1))
        noflex = np.zeros((N, 1))
        
        # Design matrix Z
        z = np.hstack([inc_vals, fulltime, flex, nofulltime, noflex])
        # transform to pandas dataframe
        df = pd.DataFrame(z, columns= ['INC', 'FULL', 'FLEX', 'NOFULL', 'NOFLEX'])

        
    return df

def print_regression_results(coefs_time, coefs_wait):
    names = [
        "Intercept",
        "INC",
        "FULL",
        "FLEX",
        "INC × FULL",
        "INC × FLEX",
        "FULL × FLEX"
    ]

    print("\n=== Regression on pred_vots ===")
    for name, coef in zip(names, coefs_time):
        print(f"{name:15s} : {coef: .6f}")

    print("\n=== Regression on pred_vowts ===")
    for name, coef in zip(names, coefs_wait):
        print(f"{name:15s} : {coef: .6f}")
        
def regress(pred_vots, pred_vowts, test_df):
    # Extract columns as numpy arrays
    inc  = test_df['INC'].values.reshape(-1, 1)
    full = test_df['FULL'].values.reshape(-1, 1)
    flex = test_df['FLEX'].values.reshape(-1, 1)

    # Interaction terms
    inc_full = inc * full
    inc_flex = inc * flex
    full_flex = full * flex

    # Design matrix Z
    Z = np.hstack([inc, full, flex, inc_full, inc_flex, full_flex])

    # Fit regressions
    reg_vot  = LinearRegression().fit(Z, pred_vots)
    reg_vowt = LinearRegression().fit(Z, pred_vowts)

    # Collect coefficients (intercept first)
    coefs_time = [reg_vot.intercept_] + list(reg_vot.coef_)
    coefs_wait = [reg_vowt.intercept_] + list(reg_vowt.coef_)

    return coefs_time, coefs_wait


def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def mabse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_pred - y_true))


def relative_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_pred - y_true) / y_true))

def signed_relative_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (y_pred - y_true) / y_true

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

def plotVOT(pred_vots, true_vots, x_values, legend_arr, result_path, simulate = False):
    fg, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
    if simulate == False:
        sort_idx = np.argsort(x_values)
        x_sorted = x_values[sort_idx]
        pred_sorted = pred_vots[sort_idx]
        ax1.plot(x_sorted, pred_sorted)
        true_sorted = true_vots[sort_idx]
        ax2.plot(x_sorted, true_sorted)
    else:    
        for vot in pred_vots:
            ax1.plot(x_values, vot)
        for v in true_vots:
            ax2.plot(x_values, v)
    ax1.set_title("Multiplicative")
    ax2.set_title("True")
    legend_labels = [', '.join(row) for row in legend_arr]
    ax1.legend(legend_labels)
    ax2.legend(legend_labels)
    ax1.set_xlabel("income ($ per hour)", fontsize=12)
    ax1.set_ylabel("value of time ($ per hour)", fontsize=12)
    ax2.set_xlabel("income ($ per hour)", fontsize=12)
    ax2.set_ylabel("value of time ($ per hour)", fontsize=12)

    os.makedirs(result_path, exist_ok=True)
    fg.savefig(result_path + "/" + "VOT_vs_inc.png", dpi=250)
    plt.close()


def plotVOWT(pred_vowts, true_vowts, x_values, legend_arr, result_path, simulate = False):
    fg, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
    if simulate == False:
        sort_idx = np.argsort(x_values)
        x_sorted = x_values[sort_idx]
        pred_sorted = pred_vowts[sort_idx]
        ax1.plot(x_sorted, pred_sorted)
        true_sorted = true_vowts[sort_idx]
        ax2.plot(x_sorted, true_sorted)
    else:
        for vowt in pred_vowts:
            ax1.plot(x_values, vowt)
        for v in true_vowts:
            ax2.plot(x_values, v)
    ax1.set_title("Multiplicative")
    ax2.set_title("True")
    legend_labels = [', '.join(row) for row in legend_arr]
    ax1.legend(legend_labels)
    ax2.legend(legend_labels)
    ax1.set_xlabel("income ($ per hour)", fontsize=12)
    ax1.set_ylabel("value of waiting time ($ per hour)", fontsize=12)
    ax2.set_xlabel("income ($ per hour)", fontsize=12)
    ax2.set_ylabel("value of waiting time ($ per hour)", fontsize=12)

    os.makedirs(result_path, exist_ok=True)
    fg.savefig(result_path + "/" + "VOWT_vs_inc.png", dpi=250)
    plt.close()
    
       

if __name__ == "__main__":
    # === Load data ===
    with open("results_one_run.pkl", "rb") as f:
        results_summary = pickle.load(f)

    # Extract relevant data
    pred_coefs_time = results_summary["coefs_time"]
    pred_coefs_wait = results_summary["coefs_wait"]
    sim_pred_vots_list = results_summary["sim_vots"]
    sim_pred_vowts_list = results_summary["sim_vowts"]
    error_coefs = results_summary["error_coefs"]
    error_vots = results_summary["error_vots"]
    error_vowts = results_summary["error_vowts"]
    accuracy = results_summary["accuracy"]
    ll_train = results_summary["ll_train"]
    ll_test = results_summary["ll_test"]

    print("\n----Outputs and Performance Metrics----")
    # Compute performance metrics
    print(f"Average Log Likelihood Loss (Train): {ll_train:.4f}")
    print(f"Average Log Likelihood Loss (Test): {ll_test:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print_regression_results(pred_coefs_time, pred_coefs_wait)
    print("\n=== Error of coefs ===")
    for key, value in error_coefs.items():
        print(f"{key}: {value:.6f}")
    print("\n=== Error of vots ===")
    for key, value in error_vots.items():
        print(f"{key}: {value:.6f}")
    print("\n=== Error of vowts ===")
    for key, value in error_vowts.items():
        print(f"{key}: {value:.6f}")    
    

    sim_df = synthetic_sociodemo(N=50, p1='deterministic')
    sim_true_vots_list = []
    sim_true_vowts_list = []
    
    base_vars = ['INC', 'FULL', 'FLEX']
    base_var_sets = [
            ['INC', 'FULL', 'FLEX'],
            ['INC', 'FULL', 'NOFLEX'],
            ['INC', 'NOFULL', 'FLEX'],
            ['INC', 'NOFULL', 'NOFLEX']
        ]

    interaction_pairs = [(0, 1), (0, 2), (1, 2)]
    true_coefs_time = [-0.1,-0.5,-0.1,0.05,-0.2,0.05,0.1]
    true_coefs_wait = [-0.2,-0.8,-0.3,0.1,-0.3,0.08,0.3]
    # Loop through each configuration
    for base_vars in base_var_sets:
        sim_true_vots, sim_true_vowts = value_of_x(
            df=sim_df,
            coef_time=true_coefs_time,
            coef_wait=true_coefs_wait,
            base_vars=base_vars,
            interactions=interaction_pairs
        )
        
        sim_true_vots_list.append(-sim_true_vots*60)
        sim_true_vowts_list.append(-sim_true_vowts*60) 
    save_dir = "results_plots/simulation_one_run" 
    os.makedirs(save_dir, exist_ok=True)  
    inc = np.asarray(sim_df['INC'])
    plotVOT(sim_pred_vots_list, sim_true_vots_list, (inc * 60),  base_var_sets, save_dir, simulate = True)
    plotVOWT(sim_pred_vowts_list, sim_true_vowts_list, (inc * 60),  base_var_sets, save_dir, simulate = True)
