

import pandas as pd
from biogeme.database import Database
from biogeme.expressions import Variable, Beta, exp, log
from biogeme.models import loglogit
from biogeme.biogeme import BIOGEME
import os
import numpy as np
from sklearn.model_selection import train_test_split
from biogeme.results_processing import get_pandas_estimated_parameters
from data_transform import pkl_to_dataframe
from sklearn.metrics import  f1_score

filename = 'train' 
# Load train dataset
X_train, Z_train, y_train, _, av_train_data = pkl_to_dataframe(filename)

filename = 'test' 
# Load test dataset
X_test, Z_test, y_test, _, av_test= pkl_to_dataframe(filename)

train_df = pd.concat([Z_train,X_train, y_train], axis=1)
test_df = pd.concat([Z_test,X_test, y_test], axis=1)

train_df['av_train'] = av_train_data[:, 0]
train_df['av_sm'] = av_train_data[:, 1]
train_df['av_car'] = av_train_data[:, 2]

test_df['av_train'] = av_test[:, 0]
test_df['av_sm'] = av_test[:, 1]
test_df['av_car'] = av_test[:, 2]

# Create a Biogeme Database using only the training set
database = Database('MNL_biogeme', train_df)

# Define variables
TRAIN_TT = Variable('TRAIN_TT')
TRAIN_HE = Variable('TRAIN_HE')
TRAIN_CO = Variable('TRAIN_CO')

SM_TT = Variable('SM_TT')
SM_HE = Variable('SM_HE')
SM_SEATS = Variable('SM_SEATS')
SM_CO = Variable('SM_CO')

CAR_TT = Variable('CAR_TT')
CAR_CO = Variable('CAR_CO')

GA = Variable('GA_1')
AGE1 = Variable('AGE_1')
AGE2 = Variable('AGE_2')
AGE3 = Variable('AGE_3')
AGE4 = Variable('AGE_4')
LUGGAGE1 = Variable('LUGGAGE_1')
LUGGAGE2 = Variable('LUGGAGE_2')


Choice = Variable('CHOICE')

# Define parameters
ASC_0 = Beta('ASC_0', 0, None, None, 0)
ASC_1 = Beta('ASC_1', 0, None, None, 0)
ASC_2 = Beta('ASC_2', 0, None, None, 1)


# Time parameters
B_time_0 = Beta('B_time_0', 0, None, None, 0)
B_time_1 = Beta('B_time_1', 0, None, None, 0)
B_time_2 = Beta('B_time_2', 0, None, None, 0)

# Headway parameters
B_headway_0 = Beta('B_headway_0', 0, None, None, 0)
B_headway_1 = Beta('B_headway_1', 0, None, None, 0)

# Seats parameter
B_seats_1 = Beta('B_seats_1', 0, None, None, 0)

# --------------------------
# Utility functions
# --------------------------
utility_train = ASC_0 + B_time_0 * TRAIN_TT + B_headway_0 * TRAIN_HE - TRAIN_CO
utility_sm = ASC_1 + B_time_1 * SM_TT + B_headway_1 * SM_HE + B_seats_1 * SM_SEATS - SM_CO
utility_car = ASC_2 + B_time_2 * CAR_TT - CAR_CO

utilities = {0: utility_train, 1:utility_sm, 2: utility_car}

availability = {
    0: Variable('av_train'),
    1: Variable('av_sm'),
    2: Variable('av_car'),
}

# Log-likelihood function
log_choice_probability = loglogit(utilities, availability, Choice)

# Estimate the model using training data
biogeme_object = BIOGEME(database, log_choice_probability)
biogeme_object.model_name = 'MNL_biogeme_swissmetro'

results = biogeme_object.estimate()

# Print model summary
print(results.short_summary())

# Extract estimated parameters
#parameters = results.getEstimatedParameters()
#betas = results.getBetaValues()
parameters = get_pandas_estimated_parameters(estimation_results=results)
print(parameters)
betas = parameters.set_index('Name')['Value'].to_dict()


# --------------------------
# **PREDICTION STEP**
# --------------------------


# Use test data for predictions
test_df['V_train'] = (
    betas['ASC_0']
    + betas['B_time_0'] * test_df['TRAIN_TT']
    + betas['B_headway_0'] * test_df['TRAIN_HE']
    - test_df['TRAIN_CO']
)

test_df['V_sm'] = (
    betas['ASC_1']
    + betas['B_time_1'] * test_df['SM_TT'] 
    + betas['B_headway_1'] * test_df['SM_HE']
    + betas['B_seats_1'] * test_df['SM_SEATS']
    - test_df['SM_CO']
)

test_df['V_car'] = (
    betas['B_time_2'] * test_df['CAR_TT'] 
    - test_df['CAR_CO']
)



# Compute exponentials
test_df['exp_V_train'] = np.exp(test_df['V_train'])* av_test[:,0]
test_df['exp_V_sm'] = np.exp(test_df['V_sm'])* av_test[:,1]
test_df['exp_V_car'] = np.exp(test_df['V_car'])* av_test[:,2]


# Compute probabilities
denom_sum = (test_df['exp_V_train'] + test_df['exp_V_sm'] + test_df['exp_V_car'])
test_df['P_train'] = test_df['exp_V_train'] / denom_sum
test_df['P_sm'] = test_df['exp_V_sm'] / denom_sum
test_df['P_car'] = test_df['exp_V_car'] / denom_sum

# Determine predicted choice
P = pd.concat([test_df['P_train'] , test_df['P_sm'], test_df['P_car']], axis=1)
test_df['predicted_choice'] = np.argmax(P, axis=1)

# Compare with actual choice
test_df['correct_prediction'] = (test_df['predicted_choice'] == (test_df['CHOICE'])).astype(int)

# Compute accuracy
accuracy = test_df['correct_prediction'].mean() * 100
error_rate = 100 - accuracy

print(f"Prediction Accuracy: {accuracy:.2f}%")
print(f"Prediction Error Rate: {error_rate:.2f}%")

# Create a Biogeme Database using on the test set
database_test = Database('MNL_biogeme_test', test_df)

availability = {
    0: Variable('av_train'),
    1: Variable('av_sm'),
    2: Variable('av_car'),
}

# Log-likelihood function
prob_train = loglogit(utilities, availability, 0)
prob_sm = loglogit(utilities, availability, 1)
prob_car = loglogit(utilities, availability, 2)

logprob = loglogit(utilities, availability, Choice)

biogeme_sim  = BIOGEME(database_test, logprob)
sim_results = biogeme_sim.simulate(results.get_beta_values())

test_loglikelihood = sim_results['log_like'].sum()
print('Test log-likelihood:', test_loglikelihood)
avg_ll = test_loglikelihood / len(sim_results)
print('Average test LL:', avg_ll)

f1_macro = f1_score(test_df['CHOICE'], test_df['predicted_choice'], average='macro')*100
f1_weighted = f1_score(test_df['CHOICE'], test_df['predicted_choice'], average='weighted')*100

print(f"Prediction F1-score (macro): {f1_macro:.2f}%")
print(f"Prediction F1-score (weighted): {f1_weighted:.2f}%")