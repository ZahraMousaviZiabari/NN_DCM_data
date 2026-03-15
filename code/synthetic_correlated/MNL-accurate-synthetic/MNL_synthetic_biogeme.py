from matplotlib import pyplot as plt
import pandas as pd
from biogeme.database import Database
from biogeme.expressions import Variable, Beta, exp, log
from biogeme.models import loglogit
from biogeme.biogeme import BIOGEME
import os
import numpy as np
from sklearn.model_selection import train_test_split
from biogeme.results_processing import get_pandas_estimated_parameters
#import data_utils

def read_data(filename) -> pd.DataFrame:
    """Read the data from file"""
    module_dir = os.path.dirname(__file__)  # Get the directory of the current file
    data_file_path = os.path.join(module_dir, 'data', filename)  
    
    # Read the data
    df = pd.read_csv(data_file_path)
    return df

filename = 'train_10k_rho_0.6_biogeme.csv' #train_10k_biogeme.csv  train100k_biogeme.csv
# Load train dataset
train_df = read_data(filename)

filename = 'test_10k_rho_0.6_biogeme.csv' #test_10k_biogeme.csv  test100k_biogeme.csv
# Load test dataset
test_df = read_data(filename)

# Split data into training (80%) and testing (20%)
#train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create a Biogeme Database using only the training set
database = Database('MNL_biogeme', train_df)

# Define variables
CAR_WT = Variable('CAR_WT')
CAR_TT = Variable('CAR_TT')
CAR_CO = Variable('CAR_CO')

TRAIN_WT = Variable('TRAIN_WT')
TRAIN_TT = Variable('TRAIN_TT')
TRAIN_CO = Variable('TRAIN_CO')


INC = Variable('INC')
FULL = Variable('FULL')
FLEX = Variable('FLEX')


Choice = Variable('CHOICE')

# Define parameters
ASC_0 = Beta('ASC_0', 0, None, None, 1)
ASC_1 = Beta('ASC_1', 0, None, None, 0)


# Time parameters (heterogeneity)
B_time0_0 = Beta('B_time0_0', 0, None, None, 0)
#B_time0_1 = Beta('B_time0_1', 0, None, None, 0)
B_time_inc = Beta('B_time_inc', 0, None, None, 0)
B_time_full = Beta('B_time_full', 0, None, None, 0)
B_time_flex = Beta('B_time_flex', 0, None, None, 0)

B_time_inc_full = Beta('B_time_inc_full', 0, None, None, 0)
B_time_inc_flex = Beta('B_time_inc_flex', 0, None, None, 0)
B_time_full_flex = Beta('B_time_full_flex', 0, None, None, 0)

# Wait parameters (heterogeneity)
B_wait0_0 = Beta('B_wait0_0', 0, None, None, 0)
#B_wait0_1 = Beta('B_wait0_1', 0, None, None, 0)
B_wait_inc = Beta('B_wait_inc', 0, None, None, 0)
B_wait_full = Beta('B_wait_full', 0, None, None, 0)
B_wait_flex = Beta('B_wait_flex', 0, None, None, 0)

B_wait_inc_full = Beta('B_wait_inc_full', 0, None, None, 0)
B_wait_inc_flex = Beta('B_wait_inc_flex', 0, None, None, 0)
B_wait_full_flex = Beta('B_wait_full_flex', 0, None, None, 0)



# --------------------------
# Marginal utility definitions
# --------------------------
mu_time = (B_time0_0
           + B_time_inc * INC
          + B_time_full * FULL
           + B_time_flex * FLEX
           + B_time_inc_full * INC * FULL
           + B_time_inc_flex * INC * FLEX
           + B_time_full_flex * FULL * FLEX

)
mu_wait = (B_wait0_0
          + B_wait_inc *  INC
           + B_wait_full * FULL 
           + B_wait_flex * FLEX
           + B_wait_inc_full * INC * FULL
           + B_wait_inc_flex * INC * FLEX
           + B_wait_full_flex * FULL * FLEX
)


# --------------------------
# Utility functions
# --------------------------
utility_train = ASC_0 + mu_time * TRAIN_TT + mu_wait * TRAIN_WT - TRAIN_CO
utility_car = ASC_1 + mu_time * CAR_TT + mu_wait * CAR_WT - CAR_CO

utilities = {0: utility_train, 1: utility_car}

# Log-likelihood function
log_choice_probability = loglogit(utilities, None, Choice)

# Estimate the model using training data
biogeme_object = BIOGEME(database, log_choice_probability)
biogeme_object.model_name = 'MNL_biogeme'

results = biogeme_object.estimate()
train_loglikelihood = results.final_loglikelihood
avg_train_ll = train_loglikelihood / len(train_df)
# Print model summary
print(results.short_summary())
print('Average train LL:', avg_train_ll)

# Extract estimated parameters
#parameters = results.getEstimatedParameters()
#betas = results.getBetaValues()
parameters = get_pandas_estimated_parameters(estimation_results=results)
print(parameters)
betas = parameters.set_index('Name')['Value'].to_dict()


# --------------------------
# **PREDICTION STEP**
# --------------------------

test_df['mu_time'] = (
    betas['B_time0_0']
    + betas['B_time_inc'] * test_df['INC']
    + betas['B_time_full'] * test_df['FULL']
    + betas['B_time_flex'] * test_df['FLEX']
    + betas['B_time_inc_full'] * test_df['INC'] * test_df['FULL']
    + betas['B_time_inc_flex'] * test_df['INC'] * test_df['FLEX']
    + betas['B_time_full_flex'] * test_df['FULL'] * test_df['FLEX']

)

test_df['mu_wait'] = (
    betas['B_wait0_0']
    + betas['B_wait_inc'] * test_df['INC']
    + betas['B_wait_full'] * test_df['FULL']
    + betas['B_wait_flex'] * test_df['FLEX']
    + betas['B_wait_inc_full'] * test_df['INC'] * test_df['FULL']
    + betas['B_wait_inc_flex'] * test_df['INC'] * test_df['FLEX']
    + betas['B_wait_full_flex'] * test_df['FULL'] * test_df['FLEX']

)


# Use test data for predictions
test_df['V_train'] = (
 test_df['mu_time'] * test_df['TRAIN_TT']
    + test_df['mu_wait'] * test_df['TRAIN_WT']
    - test_df['TRAIN_CO']
)

test_df['V_car'] = (
    betas['ASC_1']
    + test_df['mu_time'] * test_df['CAR_TT'] 
    + test_df['mu_wait'] * test_df['CAR_WT']
    - test_df['CAR_CO']
)





# Compute exponentials
test_df['exp_V_car'] = np.exp(test_df['V_car'])
test_df['exp_V_train'] = np.exp(test_df['V_train'])

# Compute probabilities
test_df['P_car'] = test_df['exp_V_car'] / (test_df['exp_V_car'] + test_df['exp_V_train'])
test_df['P_train'] = test_df['exp_V_train'] / (test_df['exp_V_car'] + test_df['exp_V_train'])

# Determine predicted choice
test_df['predicted_choice'] = np.where(test_df['P_train'] > test_df['P_car'], 0, 1)

# Compare with actual choice
test_df['correct_prediction'] = (test_df['predicted_choice'] == test_df['CHOICE']).astype(int)

# Compute accuracy
accuracy = test_df['correct_prediction'].mean() * 100
error_rate = 100 - accuracy

print(f"Prediction Accuracy: {accuracy:.2f}%")
print(f"Prediction Error Rate: {error_rate:.2f}%")

# Create a Biogeme Database using on the test set
database_test = Database('MNL_biogeme_test', test_df)

# Log-likelihood function
logprob = loglogit(utilities, None, Choice)

biogeme_sim  = BIOGEME(database_test, logprob)
sim_results = biogeme_sim.simulate(results.get_beta_values())

test_loglikelihood = sim_results['log_like'].sum()
print('Test log-likelihood:', test_loglikelihood)
avg_test_ll = test_loglikelihood / len(sim_results)
print('Average test LL:', avg_test_ll)


# ============== Simulation ==================
n = 50

sim_data_1 = pd.DataFrame({
    "INC": np.linspace(0, 1, n),
    "FULL": np.ones(n),
    "FLEX": np.ones(n)
})
sim_data_2 = pd.DataFrame({
    "INC": np.linspace(0, 1, n),
    "FULL": np.ones(n),
    "FLEX": np.zeros(n)
})
sim_data_3 = pd.DataFrame({
    "INC": np.linspace(0, 1, n),
    "FULL": np.zeros(n),
    "FLEX": np.ones(n)
})
sim_data_4 = pd.DataFrame({
    "INC": np.linspace(0, 1, n),
    "FULL": np.zeros(n),
    "FLEX": np.zeros(n)
})

# ============= simulated beta parameters ==========
#βTT_1
sim_beta_TT_1 = (
    betas['B_time0_0']
    + betas['B_time_inc'] * sim_data_1['INC']
    + betas['B_time_full'] * sim_data_1['FULL']
    + betas['B_time_flex'] * sim_data_1['FLEX']
    + betas['B_time_inc_full'] * sim_data_1['INC'] * sim_data_1['FULL']
    + betas['B_time_inc_flex'] * sim_data_1['INC'] * sim_data_1['FLEX']
    + betas['B_time_full_flex'] * sim_data_1['FULL'] * sim_data_1['FLEX']
)

#βWT_1
sim_beta_WT_1 = (
    betas['B_wait0_0']
    + betas['B_wait_inc'] * sim_data_1['INC']
    + betas['B_wait_full'] * sim_data_1['FULL']
    + betas['B_wait_flex'] * sim_data_1['FLEX']
    + betas['B_wait_inc_full'] * sim_data_1['INC'] * sim_data_1['FULL']
    + betas['B_wait_inc_flex'] * sim_data_1['INC'] * sim_data_1['FLEX']
    + betas['B_wait_full_flex'] * sim_data_1['FULL'] * sim_data_1['FLEX']
)
#βTT_2
sim_beta_TT_2 = (
    betas['B_time0_0']
    + betas['B_time_inc'] * sim_data_2['INC']
    + betas['B_time_full'] * sim_data_2['FULL']
    + betas['B_time_flex'] * sim_data_2['FLEX']
    + betas['B_time_inc_full'] * sim_data_2['INC'] * sim_data_2['FULL']
    + betas['B_time_inc_flex'] * sim_data_2['INC'] * sim_data_2['FLEX']
    + betas['B_time_full_flex'] * sim_data_2['FULL'] * sim_data_2['FLEX']
)
#βWT_2
sim_beta_WT_2 = (
    betas['B_wait0_0']
    + betas['B_wait_inc'] * sim_data_2['INC']
    + betas['B_wait_full'] * sim_data_2['FULL']
    + betas['B_wait_flex'] * sim_data_2['FLEX']
    + betas['B_wait_inc_full'] * sim_data_2['INC'] * sim_data_2['FULL']
    + betas['B_wait_inc_flex'] * sim_data_2['INC'] * sim_data_2['FLEX']
    + betas['B_wait_full_flex'] * sim_data_2['FULL'] * sim_data_2['FLEX']
)
#βTT _3
sim_beta_TT_3 = (
    betas['B_time0_0']
    + betas['B_time_inc'] * sim_data_3['INC']
    + betas['B_time_full'] * sim_data_3['FULL']
    + betas['B_time_flex'] * sim_data_3['FLEX']
    + betas['B_time_inc_full'] * sim_data_3['INC'] * sim_data_3['FULL']
    + betas['B_time_inc_flex'] * sim_data_3['INC'] * sim_data_3['FLEX']
    + betas['B_time_full_flex'] * sim_data_3['FULL'] * sim_data_3['FLEX']
)
#βWT_3
sim_beta_WT_3 = (
    betas['B_wait0_0']
    + betas['B_wait_inc'] * sim_data_3['INC']
    + betas['B_wait_full'] * sim_data_3['FULL']
    + betas['B_wait_flex'] * sim_data_3['FLEX']
    + betas['B_wait_inc_full'] * sim_data_3['INC'] * sim_data_3['FULL']
    + betas['B_wait_inc_flex'] * sim_data_3['INC'] * sim_data_3['FLEX']
    + betas['B_wait_full_flex'] * sim_data_3['FULL'] * sim_data_3['FLEX']
)
#βTT _4
sim_beta_TT_4 = (
    betas['B_time0_0']
    + betas['B_time_inc'] * sim_data_4['INC']
    + betas['B_time_full'] * sim_data_4['FULL']
    + betas['B_time_flex'] * sim_data_4['FLEX']
    + betas['B_time_inc_full'] * sim_data_4['INC'] * sim_data_4['FULL']
    + betas['B_time_inc_flex'] * sim_data_4['INC'] * sim_data_4['FLEX']
    + betas['B_time_full_flex'] * sim_data_4['FULL'] * sim_data_4['FLEX']
)
#βWT_4
sim_beta_WT_4 = (
    betas['B_wait0_0']
    + betas['B_wait_inc'] * sim_data_4['INC']
    + betas['B_wait_full'] * sim_data_4['FULL']
    + betas['B_wait_flex'] * sim_data_4['FLEX']
    + betas['B_wait_inc_full'] * sim_data_4['INC'] * sim_data_4['FULL']
    + betas['B_wait_inc_flex'] * sim_data_4['INC'] * sim_data_4['FLEX']
    + betas['B_wait_full_flex'] * sim_data_4['FULL'] * sim_data_4['FLEX']
)

vots = -pd.concat([sim_beta_TT_1, sim_beta_TT_2, sim_beta_TT_3, sim_beta_TT_4], axis=1)*60
vowts = -pd.concat([sim_beta_WT_1, sim_beta_WT_2, sim_beta_WT_3, sim_beta_WT_4], axis=1)*60
# ------------ Save ---------------
result_path = "results/"
os.makedirs(result_path, exist_ok=True)

sim_data = pd.concat({
    "vots":vots,
    "vowts": vowts,
}, axis=1)


sim_data.to_pickle(result_path + "simulated_vot_vowt.pkl")

# ------------ Plots ---------------
legend_arr = [
        ['FULL', 'FLEX'],
        ['FULL', 'NOFLEX'],
        ['NOFULL', 'FLEX'],
        ['NOFULL', 'NOFLEX']
    ]
x_values = sim_data_1["INC"].to_numpy().ravel()*60
# VoT
plt.figure(figsize=(6 ,6))
for i in range(4):
    plt.plot(x_values, vots[i])
plt.title("MNL-Acc")
legend_labels = [', '.join(row) for row in legend_arr]
plt.legend(legend_labels)
plt.xlabel("income ($ per hour)", fontsize=12)
plt.ylabel("value of time ($ per hour)", fontsize=12)
plt.savefig(result_path + "VOT_vs_inc.png", dpi=250)
plt.close()

#VoWT
plt.figure(figsize=(6 ,6))
for j in range(4):
    plt.plot(x_values, vowts[j])
plt.title("MNL-Acc")
legend_labels = [', '.join(row) for row in legend_arr]
plt.legend(legend_labels)
plt.xlabel("income ($ per hour)", fontsize=12)
plt.ylabel("value of waiting time ($ per hour)", fontsize=12)
plt.savefig(result_path + "VOWT_vs_inc.png", dpi=250)
plt.close()