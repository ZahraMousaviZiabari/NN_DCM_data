
import pandas as pd
from biogeme.database import Database
from biogeme.expressions import Variable, Beta, exp, log
from biogeme.models import loglogit
from biogeme.biogeme import BIOGEME
import os
import numpy as np
from biogeme.results_processing import get_pandas_estimated_parameters
from data_transform import pkl_to_dataframe
from sklearn.metrics import  f1_score
import matplotlib.pyplot as plt

def box_plot_VoT(plot_df):
    plt.figure(figsize=(7, 5))

    data_TT = [
        plot_df.loc[plot_df['INCOME_0'] == 1, 'beta_TT'],
        plot_df.loc[plot_df['INCOME_1'] == 1, 'beta_TT'],
        plot_df.loc[plot_df['INCOME_2'] == 1, 'beta_TT'],
    ]
    
    plt.boxplot(
        data_TT,
        labels=['Income 0', 'Income 1', 'Income 2'],
        showfliers=True
    )
    
    plt.ylabel('VOT')
    plt.title('Boxplot of VOTs by Income Group')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('Boxplot_vot_vs_inc.eps', format='eps', dpi=1000)
    
def box_plot_VoHE(plot_df):
    plt.figure(figsize=(7, 5))

    data_TT = [
        plot_df.loc[plot_df['INCOME_0'] == 1, 'beta_HE'],
        plot_df.loc[plot_df['INCOME_1'] == 1, 'beta_HE'],
        plot_df.loc[plot_df['INCOME_2'] == 1, 'beta_HE'],
    ]
    
    plt.boxplot(
        data_TT,
        labels=['Income 0', 'Income 1', 'Income 2'],
        showfliers=True
    )
    
    plt.ylabel('VOHE')
    plt.title('Boxplot of VOHEs by Income Group')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('Boxplot_vohe_vs_inc.eps', format='eps', dpi=1000)


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
INCOME0 = Variable('INCOME_0')
INCOME1 = Variable('INCOME_1')
INCOME2 = Variable('INCOME_2')
WHO1 = Variable('WHO_1')
WHO2 = Variable('WHO_2')
PURPOSE1 = Variable('PURPOSE_1')
PURPOSE2 = Variable('PURPOSE_2')
PURPOSE3 = Variable('PURPOSE_3')
MALE = Variable('MALE_1')


Choice = Variable('CHOICE')

# Define parameters
ASC_0 = Beta('ASC_0', 0, None, None, 0)
ASC_1 = Beta('ASC_1', 0, None, None, 0)
ASC_2 = Beta('ASC_2', 0, None, None, 1)


# ====== Time parameters ========
B_time0_0 = Beta('B_time0_0', 0, None, None, 0)
B_time0_1 = Beta('B_time0_1', 0, None, None, 0)
B_time0_2 = Beta('B_time0_2', 0, None, None, 0)

#====== TRAIN
B_time_age_10 = Beta('B_time_age_10', 0, None, None, 0)
B_time_age_20 = Beta('B_time_age_20', 0, None, None, 0)
B_time_age_30 = Beta('B_time_age_30', 0, None, None, 0)
B_time_age_40 = Beta('B_time_age_40', 0, None, None, 0)
B_time_lugg_10 = Beta('B_time_lugg_10', 0, None, None, 0)
B_time_lugg_20 = Beta('B_time_lugg_20', 0, None, None, 0)
B_time_ga_0 = Beta('B_time_ga_0', 0, None, None, 0)
B_time_income_00 = Beta('B_time_income_00', 0, None, None, 0)
B_time_income_10 = Beta('B_time_income_10', 0, None, None, 0)
B_time_income_20 = Beta('B_time_income_20', 0, None, None, 0)
B_time_who_10 = Beta('B_time_who_10', 0, None, None, 0)
B_time_who_20 = Beta('B_time_who_20', 0, None, None, 0)
B_time_purpose_10 = Beta('B_time_purpose_10', 0, None, None, 0)
B_time_purpose_20 = Beta('B_time_purpose_20', 0, None, None, 0)
B_time_purpose_30 = Beta('B_time_purpose_30', 0, None, None, 0)
B_time_male_0 = Beta('B_time_male_0', 0, None, None, 0)

#====== SM
B_time_age_11 = Beta('B_time_age_11', 0, None, None, 0)
B_time_age_21 = Beta('B_time_age_21', 0, None, None, 0)
B_time_age_31 = Beta('B_time_age_31', 0, None, None, 0)
B_time_age_41 = Beta('B_time_age_41', 0, None, None, 0)
B_time_lugg_11 = Beta('B_time_lugg_11', 0, None, None, 0)
B_time_lugg_21 = Beta('B_time_lugg_21', 0, None, None, 0)
B_time_ga_1 = Beta('B_time_ga_1', 0, None, None, 0)
B_time_income_01 = Beta('B_time_income_01', 0, None, None, 0)
B_time_income_11 = Beta('B_time_income_11', 0, None, None, 0)
B_time_income_21 = Beta('B_time_income_21', 0, None, None, 0)
B_time_who_11 = Beta('B_time_who_11', 0, None, None, 0)
B_time_who_21 = Beta('B_time_who_21', 0, None, None, 0)
B_time_purpose_11 = Beta('B_time_purpose_11', 0, None, None, 0)
B_time_purpose_21 = Beta('B_time_purpose_21', 0, None, None, 0)
B_time_purpose_31 = Beta('B_time_purpose_31', 0, None, None, 0)
B_time_male_1 = Beta('B_time_male_1', 0, None, None, 0)

#====== CAR
B_time_age_12 = Beta('B_time_age_12', 0, None, None, 0)
B_time_age_22 = Beta('B_time_age_22', 0, None, None, 0)
B_time_age_32 = Beta('B_time_age_32', 0, None, None, 0)
B_time_age_42 = Beta('B_time_age_42', 0, None, None, 0)
B_time_lugg_12 = Beta('B_time_lugg_12', 0, None, None, 0)
B_time_lugg_22 = Beta('B_time_lugg_22', 0, None, None, 0)
B_time_ga_2 = Beta('B_time_ga_2', 0, None, None, 0)
B_time_income_02 = Beta('B_time_income_02', 0, None, None, 0)
B_time_income_12 = Beta('B_time_income_12', 0, None, None, 0)
B_time_income_22 = Beta('B_time_income_22', 0, None, None, 0)
B_time_who_12 = Beta('B_time_who_12', 0, None, None, 0)
B_time_who_22 = Beta('B_time_who_22', 0, None, None, 0)
B_time_purpose_12 = Beta('B_time_purpose_12', 0, None, None, 0)
B_time_purpose_22 = Beta('B_time_purpose_22', 0, None, None, 0)
B_time_purpose_32 = Beta('B_time_purpose_32', 0, None, None, 0)
B_time_male_2 = Beta('B_time_male_2', 0, None, None, 0)

# ======= Headway parameters ==========
B_headway0_0 = Beta('B_headway0_0', 0, None, None, 0)
B_headway0_1 = Beta('B_headway0_1', 0, None, None, 0)

#====== TRAIN
B_headway_age_10 = Beta('B_headway_age_10', 0, None, None, 0)
B_headway_age_20 = Beta('B_headway_age_20', 0, None, None, 0)
B_headway_age_30 = Beta('B_headway_age_30', 0, None, None, 0)
B_headway_age_40 = Beta('B_headway_age_40', 0, None, None, 0)
B_headway_lugg_10 = Beta('B_headway_lugg_10', 0, None, None, 0)
B_headway_lugg_20 = Beta('B_headway_lugg_20', 0, None, None, 0)
B_headway_ga_0 = Beta('B_headway_ga_0', 0, None, None, 0)
B_headway_income_00 = Beta('B_headway_income_00', 0, None, None, 0)
B_headway_income_10 = Beta('B_headway_income_10', 0, None, None, 0)
B_headway_income_20 = Beta('B_headway_income_20', 0, None, None, 0)
B_headway_who_10 = Beta('B_headway_who_10', 0, None, None, 0)
B_headway_who_20 = Beta('B_headway_who_20', 0, None, None, 0)
B_headway_purpose_10 = Beta('B_headway_purpose_10', 0, None, None, 0)
B_headway_purpose_20 = Beta('B_headway_purpose_20', 0, None, None, 0)
B_headway_purpose_30 = Beta('B_headway_purpose_30', 0, None, None, 0)
B_headway_male_0 = Beta('B_headway_male_0', 0, None, None, 0)

#====== SM
B_headway_age_11 = Beta('B_headway_age_11', 0, None, None, 0)
B_headway_age_21 = Beta('B_headway_age_21', 0, None, None, 0)
B_headway_age_31 = Beta('B_headway_age_31', 0, None, None, 0)
B_headway_age_41 = Beta('B_headway_age_41', 0, None, None, 0)
B_headway_lugg_11 = Beta('B_headway_lugg_11', 0, None, None, 0)
B_headway_lugg_21 = Beta('B_headway_lugg_21', 0, None, None, 0)
B_headway_ga_1 = Beta('B_headway_ga_1', 0, None, None, 0)
B_headway_income_01 = Beta('B_headway_income_01', 0, None, None, 0)
B_headway_income_11 = Beta('B_headway_income_11', 0, None, None, 0)
B_headway_income_21 = Beta('B_headway_income_21', 0, None, None, 0)
B_headway_who_11 = Beta('B_headway_who_11', 0, None, None, 0)
B_headway_who_21 = Beta('B_headway_who_21', 0, None, None, 0)
B_headway_purpose_11 = Beta('B_headway_purpose_11', 0, None, None, 0)
B_headway_purpose_21 = Beta('B_headway_purpose_21', 0, None, None, 0)
B_headway_purpose_31 = Beta('B_headway_purpose_31', 0, None, None, 0)
B_headway_male_1 = Beta('B_headway_male_1', 0, None, None, 0)

# ======= Seats parameters ==========
B_seats0_1 = Beta('B_seats0_1', 0, None, None, 0)

B_seats_age_11 = Beta('B_seats_age_11', 0, None, None, 0)
B_seats_age_21 = Beta('B_seats_age_21', 0, None, None, 0)
B_seats_age_31 = Beta('B_seats_age_31', 0, None, None, 0)
B_seats_age_41 = Beta('B_seats_age_41', 0, None, None, 0)
B_seats_lugg_11 = Beta('B_seats_lugg_11', 0, None, None, 0)
B_seats_lugg_21 = Beta('B_seats_lugg_21', 0, None, None, 0)
B_seats_ga_1 = Beta('B_seats_ga_1', 0, None, None, 0)
B_seats_income_01 = Beta('B_seats_income_01', 0, None, None, 0)
B_seats_income_11 = Beta('B_seats_income_11', 0, None, None, 0)
B_seats_income_21 = Beta('B_seats_income_21', 0, None, None, 0)
B_seats_who_11 = Beta('B_seats_who_11', 0, None, None, 0)
B_seats_who_21 = Beta('B_seats_who_21', 0, None, None, 0)
B_seats_purpose_11 = Beta('B_seats_purpose_11', 0, None, None, 0)
B_seats_purpose_21 = Beta('B_seats_purpose_21', 0, None, None, 0)
B_seats_purpose_31 = Beta('B_seats_purpose_31', 0, None, None, 0)
B_seats_male_1 = Beta('B_seats_male_1', 0, None, None, 0)

# ======= Sociodemographic parameters ==========
#====== SM
B_age_11 = Beta('B_age_11', 0, None, None, 0)
B_age_21 = Beta('B_age_21', 0, None, None, 0)
B_age_31 = Beta('B_age_31', 0, None, None, 0)
B_age_41 = Beta('B_age_41', 0, None, None, 0)
B_lugg_11 = Beta('B_lugg_11', 0, None, None, 0)
B_lugg_21 = Beta('B_lugg_21', 0, None, None, 0)
B_ga_1 = Beta('B_ga_1', 0, None, None, 0)
B_income_01 = Beta('B_income_01', 0, None, None, 0)
B_income_11 = Beta('B_income_11', 0, None, None, 0)
B_income_21 = Beta('B_income_21', 0, None, None, 0)
B_who_11 = Beta('B_who_11', 0, None, None, 0)
B_who_21 = Beta('B_who_21', 0, None, None, 0)
B_purpose_11 = Beta('B_purpose_11', 0, None, None, 0)
B_purpose_21 = Beta('B_purpose_21', 0, None, None, 0)
B_purpose_31 = Beta('B_purpose_31', 0, None, None, 0)
B_male_1 = Beta('B_male_1', 0, None, None, 0)

#====== CAR
B_age_12 = Beta('B_age_12', 0, None, None, 0)
B_age_22 = Beta('B_age_22', 0, None, None, 0)
B_age_32 = Beta('B_age_32', 0, None, None, 0)
B_age_42 = Beta('B_age_42', 0, None, None, 0)
B_lugg_12 = Beta('B_lugg_12', 0, None, None, 0)
B_lugg_22 = Beta('B_lugg_22', 0, None, None, 0)
B_ga_2 = Beta('B_ga_2', 0, None, None, 0)
B_income_02 = Beta('B_income_02', 0, None, None, 0)
B_income_12 = Beta('B_income_12', 0, None, None, 0)
B_income_22 = Beta('B_income_22', 0, None, None, 0)
B_who_12 = Beta('B_who_12', 0, None, None, 0)
B_who_22 = Beta('B_who_22', 0, None, None, 0)
B_purpose_12 = Beta('B_purpose_12', 0, None, None, 0)
B_purpose_22 = Beta('B_purpose_22', 0, None, None, 0)
B_purpose_32 = Beta('B_purpose_32', 0, None, None, 0)
B_male_2 = Beta('B_male_2', 0, None, None, 0)
# --------------------------
# Marginal utility definitions
# --------------------------
#====== TRAIN
mu_time_0 = (B_time0_0
           + B_time_age_10 * AGE1
           + B_time_age_20 * AGE2
           + B_time_age_30 * AGE3
           + B_time_age_40 * AGE4
           + B_time_lugg_10 * LUGGAGE1
           + B_time_lugg_20 * LUGGAGE2
           + B_time_ga_0 * GA
           + B_time_income_00 * INCOME0
           + B_time_income_10 * INCOME1
           + B_time_income_20 * INCOME2
           + B_time_who_10 * WHO1
           + B_time_who_20 * WHO2
           + B_time_purpose_10 * PURPOSE1
           + B_time_purpose_20 * PURPOSE2
           + B_time_purpose_30 * PURPOSE3
           + B_time_male_0 * MALE

)
mu_headway_0 = (B_headway0_0
          + B_headway_age_10 * AGE1
           + B_headway_age_20 * AGE2 
           + B_headway_age_30 * AGE3
           + B_headway_age_40 * AGE4
           + B_headway_lugg_10 * LUGGAGE1
           + B_headway_lugg_20 * LUGGAGE2
           + B_headway_ga_0 * GA
           + B_headway_income_00 * INCOME0
           + B_headway_income_10 * INCOME1
           + B_headway_income_20 * INCOME2
           + B_headway_who_10 * WHO1
           + B_headway_who_20 * WHO2
           + B_headway_purpose_10 * PURPOSE1
           + B_headway_purpose_20 * PURPOSE2
           + B_headway_purpose_30 * PURPOSE3
           + B_headway_male_0 * MALE
)

#========= SM

mu_time_1 = (B_time0_1
           + B_time_age_11 * AGE1
           + B_time_age_21 * AGE2
           + B_time_age_31 * AGE3
           + B_time_age_41 * AGE4
           + B_time_lugg_11 * LUGGAGE1
           + B_time_lugg_21 * LUGGAGE2
           + B_time_ga_1 * GA
           + B_time_income_01 * INCOME0
           + B_time_income_11 * INCOME1
           + B_time_income_21 * INCOME2
           + B_time_who_11 * WHO1
           + B_time_who_21 * WHO2
           + B_time_purpose_11 * PURPOSE1
           + B_time_purpose_21 * PURPOSE2
           + B_time_purpose_31 * PURPOSE3
           + B_time_male_1 * MALE

)
mu_headway_1 = (B_headway0_1
          + B_headway_age_11 * AGE1
           + B_headway_age_21 * AGE2 
           + B_headway_age_31 * AGE3
           + B_headway_age_41 * AGE4
           + B_headway_lugg_11 * LUGGAGE1
           + B_headway_lugg_21 * LUGGAGE2
           + B_headway_ga_1 * GA
           + B_headway_income_01 * INCOME0
           + B_headway_income_11 * INCOME1
           + B_headway_income_21 * INCOME2
           + B_headway_who_11 * WHO1
           + B_headway_who_21 * WHO2
           + B_headway_purpose_11 * PURPOSE1
           + B_headway_purpose_21 * PURPOSE2
           + B_headway_purpose_31 * PURPOSE3
           + B_headway_male_1 * MALE
)

mu_seats_1 = (B_seats0_1
          + B_seats_age_11 * AGE1
           + B_seats_age_21 * AGE2 
           + B_seats_age_31 * AGE3
           + B_seats_age_41 * AGE4
           + B_seats_lugg_11 * LUGGAGE1
           + B_seats_lugg_21 * LUGGAGE2
           + B_seats_ga_1 * GA
           + B_seats_income_01 * INCOME0
           + B_seats_income_11 * INCOME1
           + B_seats_income_21 * INCOME2
           + B_seats_who_11 * WHO1
           + B_seats_who_21 * WHO2
           + B_seats_purpose_11 * PURPOSE1
           + B_seats_purpose_21 * PURPOSE2
           + B_seats_purpose_31 * PURPOSE3
           + B_seats_male_1 * MALE
)

#============== CAR

mu_time_2 = (B_time0_2
           + B_time_age_12 * AGE1
           + B_time_age_22 * AGE2
           + B_time_age_32 * AGE3
           + B_time_age_42 * AGE4
           + B_time_lugg_12 * LUGGAGE1
           + B_time_lugg_22 * LUGGAGE2
           + B_time_ga_2 * GA
           + B_time_income_02 * INCOME0
           + B_time_income_12 * INCOME1
           + B_time_income_22 * INCOME2
           + B_time_who_12 * WHO1
           + B_time_who_22 * WHO2
           + B_time_purpose_12 * PURPOSE1
           + B_time_purpose_22 * PURPOSE2
           + B_time_purpose_32 * PURPOSE3
           + B_time_male_2 * MALE

)


# --------------------------
# Utility functions
# --------------------------
utility_train = ASC_0 + mu_time_0 * TRAIN_TT + mu_headway_0 * TRAIN_HE - TRAIN_CO
utility_sm = ASC_1 + mu_time_1 * SM_TT + mu_headway_1 * SM_HE + mu_seats_1 * SM_SEATS - SM_CO \
    + B_age_11 * AGE1 + B_age_21 * AGE2 + B_age_31 * AGE3 + B_age_41 * AGE4 + B_lugg_11 * LUGGAGE1\
    + B_lugg_21 * LUGGAGE2 + B_ga_1 * GA + B_income_01 * INCOME0 + B_income_11 * INCOME1 + B_income_21 * INCOME2\
    + B_who_11 * WHO1 + B_who_21 * WHO2 + B_purpose_11 * PURPOSE1 + B_purpose_21 * PURPOSE2 \
    + B_purpose_31 * PURPOSE3 + B_male_1 * MALE
utility_car = ASC_2 + mu_time_2 * CAR_TT - CAR_CO\
    + B_age_12 * AGE1 + B_age_22 * AGE2 + B_age_32 * AGE3 + B_age_42 * AGE4 + B_lugg_12 * LUGGAGE1\
    + B_lugg_22 * LUGGAGE2 + B_ga_2 * GA + B_income_02 * INCOME0 + B_income_12 * INCOME1 + B_income_22 * INCOME2\
    + B_who_12 * WHO1 + B_who_22 * WHO2 + B_purpose_12 * PURPOSE1 + B_purpose_22 * PURPOSE2 \
    + B_purpose_32 * PURPOSE3 + B_male_2 * MALE

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

# ============= beta parameters ==========
#βTT-TRAIN
beta_TT_TRAIN = (
    betas['B_time0_0']
    + betas['B_time_age_10'] * train_df['AGE_1']
    + betas['B_time_age_20'] * train_df['AGE_2']
    + betas['B_time_age_30'] * train_df['AGE_3']
    + betas['B_time_age_40'] * train_df['AGE_4']
    + betas['B_time_lugg_10'] * train_df['LUGGAGE_1']
    + betas['B_time_lugg_20'] * train_df['LUGGAGE_2']
    + betas['B_time_ga_0'] * train_df['GA_1']
    + betas['B_time_income_00'] * train_df['INCOME_0']
    + betas['B_time_income_10'] * train_df['INCOME_1']
    + betas['B_time_income_20'] * train_df['INCOME_2']
    + betas['B_time_who_10'] * train_df['WHO_1']
    + betas['B_time_who_20'] * train_df['WHO_2']
    + betas['B_time_purpose_10'] * train_df['PURPOSE_1']
    + betas['B_time_purpose_20'] * train_df['PURPOSE_2']
    + betas['B_time_purpose_30'] * train_df['PURPOSE_3']
    + betas['B_time_male_0'] * train_df['MALE_1']
)

#βHE-TRAIN
beta_HE_TRAIN = (
    betas['B_headway0_0']
    + betas['B_headway_age_10'] * train_df['AGE_1']
    + betas['B_headway_age_20'] * train_df['AGE_2']
    + betas['B_headway_age_30'] * train_df['AGE_3']
    + betas['B_headway_age_40'] * train_df['AGE_4']
    + betas['B_headway_lugg_10'] * train_df['LUGGAGE_1']
    + betas['B_headway_lugg_20'] * train_df['LUGGAGE_2']
    + betas['B_headway_ga_0'] * train_df['GA_1']
    + betas['B_headway_income_00'] * train_df['INCOME_0']
    + betas['B_headway_income_10'] * train_df['INCOME_1']
    + betas['B_headway_income_20'] * train_df['INCOME_2']
    + betas['B_headway_who_10'] * train_df['WHO_1']
    + betas['B_headway_who_20'] * train_df['WHO_2']
    + betas['B_headway_purpose_10'] * train_df['PURPOSE_1']
    + betas['B_headway_purpose_20'] * train_df['PURPOSE_2']
    + betas['B_headway_purpose_30'] * train_df['PURPOSE_3']
    + betas['B_headway_male_0'] * train_df['MALE_1']
)

#βTT-SM
beta_TT_SM = (
    betas['B_time0_1']
    + betas['B_time_age_11'] * train_df['AGE_1']
    + betas['B_time_age_21'] * train_df['AGE_2']
    + betas['B_time_age_31'] * train_df['AGE_3']
    + betas['B_time_age_41'] * train_df['AGE_4']
    + betas['B_time_lugg_11'] * train_df['LUGGAGE_1']
    + betas['B_time_lugg_21'] * train_df['LUGGAGE_2']
    + betas['B_time_ga_1'] * train_df['GA_1']
    + betas['B_time_income_01'] * train_df['INCOME_0']
    + betas['B_time_income_11'] * train_df['INCOME_1']
    + betas['B_time_income_21'] * train_df['INCOME_2']
    + betas['B_time_who_11'] * train_df['WHO_1']
    + betas['B_time_who_21'] * train_df['WHO_2']
    + betas['B_time_purpose_11'] * train_df['PURPOSE_1']
    + betas['B_time_purpose_21'] * train_df['PURPOSE_2']
    + betas['B_time_purpose_31'] * train_df['PURPOSE_3']
    + betas['B_time_male_1'] * train_df['MALE_1']
)

#βHE-SM
beta_HE_SM = (
    betas['B_headway0_1']
    + betas['B_headway_age_11'] * train_df['AGE_1']
    + betas['B_headway_age_21'] * train_df['AGE_2']
    + betas['B_headway_age_31'] * train_df['AGE_3']
    + betas['B_headway_age_41'] * train_df['AGE_4']
    + betas['B_headway_lugg_11'] * train_df['LUGGAGE_1']
    + betas['B_headway_lugg_21'] * train_df['LUGGAGE_2']
    + betas['B_headway_ga_1'] * train_df['GA_1']
    + betas['B_headway_income_01'] * train_df['INCOME_0']
    + betas['B_headway_income_11'] * train_df['INCOME_1']
    + betas['B_headway_income_21'] * train_df['INCOME_2']
    + betas['B_headway_who_11'] * train_df['WHO_1']
    + betas['B_headway_who_21'] * train_df['WHO_2']
    + betas['B_headway_purpose_11'] * train_df['PURPOSE_1']
    + betas['B_headway_purpose_21'] * train_df['PURPOSE_2']
    + betas['B_headway_purpose_31'] * train_df['PURPOSE_3']
    + betas['B_headway_male_1'] * train_df['MALE_1']
)

#βSEATS-SM
beta_SEATS_SM = (
    betas['B_seats0_1']
    + betas['B_seats_age_11'] * train_df['AGE_1']
    + betas['B_seats_age_21'] * train_df['AGE_2']
    + betas['B_seats_age_31'] * train_df['AGE_3']
    + betas['B_seats_age_41'] * train_df['AGE_4']
    + betas['B_seats_lugg_11'] * train_df['LUGGAGE_1']
    + betas['B_seats_lugg_21'] * train_df['LUGGAGE_2']
    + betas['B_seats_ga_1'] * train_df['GA_1']
    + betas['B_seats_income_01'] * train_df['INCOME_0']
    + betas['B_seats_income_11'] * train_df['INCOME_1']
    + betas['B_seats_income_21'] * train_df['INCOME_2']
    + betas['B_seats_who_11'] * train_df['WHO_1']
    + betas['B_seats_who_21'] * train_df['WHO_2']
    + betas['B_seats_purpose_11'] * train_df['PURPOSE_1']
    + betas['B_seats_purpose_21'] * train_df['PURPOSE_2']
    + betas['B_seats_purpose_31'] * train_df['PURPOSE_3']
    + betas['B_seats_male_1'] * train_df['MALE_1']
)

#βTT-CAR
beta_TT_CAR = (
    betas['B_time0_2']
    + betas['B_time_age_12'] * train_df['AGE_1']
    + betas['B_time_age_22'] * train_df['AGE_2']
    + betas['B_time_age_32'] * train_df['AGE_3']
    + betas['B_time_age_42'] * train_df['AGE_4']
    + betas['B_time_lugg_12'] * train_df['LUGGAGE_1']
    + betas['B_time_lugg_22'] * train_df['LUGGAGE_2']
    + betas['B_time_ga_2'] * train_df['GA_1']
    + betas['B_time_income_02'] * train_df['INCOME_0']
    + betas['B_time_income_12'] * train_df['INCOME_1']
    + betas['B_time_income_22'] * train_df['INCOME_2']
    + betas['B_time_who_12'] * train_df['WHO_1']
    + betas['B_time_who_22'] * train_df['WHO_2']
    + betas['B_time_purpose_12'] * train_df['PURPOSE_1']
    + betas['B_time_purpose_22'] * train_df['PURPOSE_2']
    + betas['B_time_purpose_32'] * train_df['PURPOSE_3']
    + betas['B_time_male_2'] * train_df['MALE_1']
)

ASC_TRAIN = betas['ASC_0']
ASC_SM = (
    betas['ASC_1']
    + betas['B_age_11'] * train_df['AGE_1']
    + betas['B_age_21'] * train_df['AGE_2']
    + betas['B_age_31'] * train_df['AGE_3']
    + betas['B_age_41'] * train_df['AGE_4']
    + betas['B_lugg_11'] * train_df['LUGGAGE_1']
    + betas['B_lugg_21'] * train_df['LUGGAGE_2']
    + betas['B_ga_1'] * train_df['GA_1']
    + betas['B_income_01'] * train_df['INCOME_0']
    + betas['B_income_11'] * train_df['INCOME_1']
    + betas['B_income_21'] * train_df['INCOME_2']
    + betas['B_who_11'] * train_df['WHO_1']
    + betas['B_who_21'] * train_df['WHO_2']
    + betas['B_purpose_11'] * train_df['PURPOSE_1']
    + betas['B_purpose_21'] * train_df['PURPOSE_2']
    + betas['B_purpose_31'] * train_df['PURPOSE_3']
    + betas['B_male_1'] * train_df['MALE_1']
)


beta_TT_TRAIN_mean = beta_TT_TRAIN.mean()
beta_HE_TRAIN_mean = beta_HE_TRAIN.mean()
beta_TT_SM_mean = beta_TT_SM.mean()
beta_HE_SM_mean = beta_HE_SM.mean()
beta_SEATS_SM_mean = beta_SEATS_SM.mean()
beta_TT_CAR_mean = beta_TT_CAR.mean()

beta_TT_TRAIN_std = beta_TT_TRAIN.std()
beta_HE_TRAIN_std = beta_HE_TRAIN.std()
beta_TT_SM_std = beta_TT_SM.std()
beta_HE_SM_std = beta_HE_SM.std()
beta_SEATS_SM_std = beta_SEATS_SM.std()
beta_TT_CAR_std = beta_TT_CAR.std()

print("==== mean beta values ====")
print("b_TT-TRAIN:", beta_TT_TRAIN_mean)
print("b_HE-TRAIN:", beta_HE_TRAIN_mean)
print("b_TT-SM:", beta_TT_SM_mean)
print("b_HE-SM:", beta_HE_SM_mean)
print("b_SEATS-SM:", beta_SEATS_SM_mean)
print("b_TT-CAR:", beta_TT_CAR_mean)

print("==== std beta values ====")
print("b_TT-TRAIN:", beta_TT_TRAIN_std)
print("b_HE-TRAIN:", beta_HE_TRAIN_std)
print("b_TT-SM:", beta_TT_SM_std)
print("b_HE-SM:", beta_HE_SM_std)
print("b_SEATS-SM:", beta_SEATS_SM_std)
print("b_TT-CAR:", beta_TT_CAR_std)
print("==== asc values ====")
print("ASC_TRAIN:", ASC_TRAIN)
print("ASC_SM:", ASC_SM.mean())

# ------------ Plots ---------------
plot_df = train_df[['INCOME_0', 'INCOME_1', 'INCOME_2']].copy()
plot_df['beta_TT'] = -beta_TT_TRAIN.values
plot_df['beta_HE'] = -beta_HE_TRAIN.values
plot_df.to_pickle("vot_vohe_traindata.pkl")
box_plot_VoT(plot_df)
box_plot_VoHE(plot_df)

# --------------------------
# **PREDICTION STEP**
# --------------------------


# Use test data for predictions
test_df['V_train'] = (
    betas['ASC_0']
    + betas['B_time0_0'] * test_df['TRAIN_TT']
    
    + betas['B_time_age_10'] * test_df['AGE_1'] * test_df['TRAIN_TT']
    + betas['B_time_age_20'] * test_df['AGE_2'] * test_df['TRAIN_TT']
    + betas['B_time_age_30'] * test_df['AGE_3'] * test_df['TRAIN_TT']
    + betas['B_time_age_40'] * test_df['AGE_4'] * test_df['TRAIN_TT']
    + betas['B_time_lugg_10'] * test_df['LUGGAGE_1'] * test_df['TRAIN_TT']
    + betas['B_time_lugg_20'] * test_df['LUGGAGE_2'] * test_df['TRAIN_TT']
    + betas['B_time_ga_0'] * test_df['GA_1'] * test_df['TRAIN_TT']
    + betas['B_time_income_00'] * test_df['INCOME_0'] * test_df['TRAIN_TT']
    + betas['B_time_income_10'] * test_df['INCOME_1'] * test_df['TRAIN_TT']
    + betas['B_time_income_20'] * test_df['INCOME_2'] * test_df['TRAIN_TT']
    + betas['B_time_who_10'] * test_df['WHO_1'] * test_df['TRAIN_TT']
    + betas['B_time_who_20'] * test_df['WHO_2'] * test_df['TRAIN_TT']
    + betas['B_time_purpose_10'] * test_df['PURPOSE_1'] * test_df['TRAIN_TT']
    + betas['B_time_purpose_20'] * test_df['PURPOSE_2'] * test_df['TRAIN_TT']
    + betas['B_time_purpose_30'] * test_df['PURPOSE_3'] * test_df['TRAIN_TT']
    + betas['B_time_male_0'] * test_df['MALE_1'] * test_df['TRAIN_TT']
    
    + betas['B_headway0_0'] * test_df['TRAIN_HE']
    + betas['B_headway_age_10'] * test_df['AGE_1'] * test_df['TRAIN_HE']
    + betas['B_headway_age_20'] * test_df['AGE_2'] * test_df['TRAIN_HE']
    + betas['B_headway_age_30'] * test_df['AGE_3'] * test_df['TRAIN_HE']
    + betas['B_headway_age_40'] * test_df['AGE_4'] * test_df['TRAIN_HE']
    + betas['B_headway_lugg_10'] * test_df['LUGGAGE_1'] * test_df['TRAIN_HE']
    + betas['B_headway_lugg_20'] * test_df['LUGGAGE_2'] * test_df['TRAIN_HE']
    + betas['B_headway_ga_0'] * test_df['GA_1'] * test_df['TRAIN_HE']
    + betas['B_headway_income_00'] * test_df['INCOME_0'] * test_df['TRAIN_HE']
    + betas['B_headway_income_10'] * test_df['INCOME_1'] * test_df['TRAIN_HE']
    + betas['B_headway_income_20'] * test_df['INCOME_2'] * test_df['TRAIN_HE']
    + betas['B_headway_who_10'] * test_df['WHO_1'] * test_df['TRAIN_HE']
    + betas['B_headway_who_20'] * test_df['WHO_2'] * test_df['TRAIN_HE']
    + betas['B_headway_purpose_10'] * test_df['PURPOSE_1'] * test_df['TRAIN_HE']
    + betas['B_headway_purpose_20'] * test_df['PURPOSE_2'] * test_df['TRAIN_HE']
    + betas['B_headway_purpose_30'] * test_df['PURPOSE_3'] * test_df['TRAIN_HE']
    + betas['B_headway_male_0'] * test_df['MALE_1'] * test_df['TRAIN_HE']
        
    - test_df['TRAIN_CO']
)


test_df['V_sm'] = (
    betas['ASC_1']
    + betas['B_time0_1'] * test_df['SM_TT']
    
    + betas['B_time_age_11'] * test_df['AGE_1'] * test_df['SM_TT']
    + betas['B_time_age_21'] * test_df['AGE_2'] * test_df['SM_TT']
    + betas['B_time_age_31'] * test_df['AGE_3'] * test_df['SM_TT']
    + betas['B_time_age_41'] * test_df['AGE_4'] * test_df['SM_TT']
    + betas['B_time_lugg_11'] * test_df['LUGGAGE_1'] * test_df['SM_TT']
    + betas['B_time_lugg_21'] * test_df['LUGGAGE_2'] * test_df['SM_TT']
    + betas['B_time_ga_1'] * test_df['GA_1'] * test_df['SM_TT']
    + betas['B_time_income_01'] * test_df['INCOME_0'] * test_df['SM_TT']
    + betas['B_time_income_11'] * test_df['INCOME_1'] * test_df['SM_TT']
    + betas['B_time_income_21'] * test_df['INCOME_2'] * test_df['SM_TT']
    + betas['B_time_who_11'] * test_df['WHO_1'] * test_df['SM_TT']
    + betas['B_time_who_21'] * test_df['WHO_2'] * test_df['SM_TT']
    + betas['B_time_purpose_11'] * test_df['PURPOSE_1'] * test_df['SM_TT']
    + betas['B_time_purpose_21'] * test_df['PURPOSE_2'] * test_df['SM_TT']
    + betas['B_time_purpose_31'] * test_df['PURPOSE_3'] * test_df['SM_TT']
    + betas['B_time_male_1'] * test_df['MALE_1'] * test_df['SM_TT']

    + betas['B_headway0_1'] * test_df['SM_HE']
    
    + betas['B_headway_age_11'] * test_df['AGE_1'] * test_df['SM_HE']
    + betas['B_headway_age_21'] * test_df['AGE_2'] * test_df['SM_HE']
    + betas['B_headway_age_31'] * test_df['AGE_3'] * test_df['SM_HE']
    + betas['B_headway_age_41'] * test_df['AGE_4'] * test_df['SM_HE']
    + betas['B_headway_lugg_11'] * test_df['LUGGAGE_1'] * test_df['SM_HE']
    + betas['B_headway_lugg_21'] * test_df['LUGGAGE_2'] * test_df['SM_HE']
    + betas['B_headway_ga_1'] * test_df['GA_1'] * test_df['SM_HE']
    + betas['B_headway_income_01'] * test_df['INCOME_0'] * test_df['SM_HE']
    + betas['B_headway_income_11'] * test_df['INCOME_1'] * test_df['SM_HE']
    + betas['B_headway_income_21'] * test_df['INCOME_2'] * test_df['SM_HE']
    + betas['B_headway_who_11'] * test_df['WHO_1'] * test_df['SM_HE']
    + betas['B_headway_who_21'] * test_df['WHO_2'] * test_df['SM_HE']
    + betas['B_headway_purpose_11'] * test_df['PURPOSE_1'] * test_df['SM_HE']
    + betas['B_headway_purpose_21'] * test_df['PURPOSE_2'] * test_df['SM_HE']
    + betas['B_headway_purpose_31'] * test_df['PURPOSE_3'] * test_df['SM_HE']
    + betas['B_headway_male_1'] * test_df['MALE_1'] * test_df['SM_HE']
    
    + betas['B_seats0_1'] * test_df['SM_SEATS']
    
    + betas['B_seats_age_11'] * test_df['AGE_1'] * test_df['SM_SEATS']
    + betas['B_seats_age_21'] * test_df['AGE_2'] * test_df['SM_SEATS']
    + betas['B_seats_age_31'] * test_df['AGE_3'] * test_df['SM_SEATS']
    + betas['B_seats_age_41'] * test_df['AGE_4'] * test_df['SM_SEATS']
    + betas['B_seats_lugg_11'] * test_df['LUGGAGE_1'] * test_df['SM_SEATS']
    + betas['B_seats_lugg_21'] * test_df['LUGGAGE_2'] * test_df['SM_SEATS']
    + betas['B_seats_ga_1'] * test_df['GA_1'] * test_df['SM_SEATS']
    + betas['B_seats_income_01'] * test_df['INCOME_0'] * test_df['SM_SEATS']
    + betas['B_seats_income_11'] * test_df['INCOME_1'] * test_df['SM_SEATS']
    + betas['B_seats_income_21'] * test_df['INCOME_2'] * test_df['SM_SEATS']
    + betas['B_seats_who_11'] * test_df['WHO_1'] * test_df['SM_SEATS']
    + betas['B_seats_who_21'] * test_df['WHO_2'] * test_df['SM_SEATS']
    + betas['B_seats_purpose_11'] * test_df['PURPOSE_1'] * test_df['SM_SEATS']
    + betas['B_seats_purpose_21'] * test_df['PURPOSE_2'] * test_df['SM_SEATS']
    + betas['B_seats_purpose_31'] * test_df['PURPOSE_3'] * test_df['SM_SEATS']
    + betas['B_seats_male_1'] * test_df['MALE_1'] * test_df['SM_SEATS']
    
    + betas['B_age_11'] * test_df['AGE_1'] 
    + betas['B_age_21'] * test_df['AGE_2'] 
    + betas['B_age_31'] * test_df['AGE_3'] 
    + betas['B_age_41'] * test_df['AGE_4'] 
    + betas['B_lugg_11'] * test_df['LUGGAGE_1']
    + betas['B_lugg_21'] * test_df['LUGGAGE_2'] 
    + betas['B_ga_1'] * test_df['GA_1'] 
    + betas['B_income_01'] * test_df['INCOME_0'] 
    + betas['B_income_11'] * test_df['INCOME_1'] 
    + betas['B_income_21'] * test_df['INCOME_2'] 
    + betas['B_who_11'] * test_df['WHO_1'] 
    + betas['B_who_21'] * test_df['WHO_2'] 
    + betas['B_purpose_11'] * test_df['PURPOSE_1'] 
    + betas['B_purpose_21'] * test_df['PURPOSE_2'] 
    + betas['B_purpose_31'] * test_df['PURPOSE_3'] 
    + betas['B_male_1'] * test_df['MALE_1'] 
    
    - test_df['SM_CO']
)

test_df['V_car'] = (
    betas['B_time0_2'] * test_df['CAR_TT']
    
    + betas['B_time_age_12'] * test_df['AGE_1'] * test_df['CAR_TT']
    + betas['B_time_age_22'] * test_df['AGE_2'] * test_df['CAR_TT']
    + betas['B_time_age_32'] * test_df['AGE_3'] * test_df['CAR_TT']
    + betas['B_time_age_42'] * test_df['AGE_4'] * test_df['CAR_TT']
    + betas['B_time_lugg_12'] * test_df['LUGGAGE_1'] * test_df['CAR_TT']
    + betas['B_time_lugg_22'] * test_df['LUGGAGE_2'] * test_df['CAR_TT']
    + betas['B_time_ga_2'] * test_df['GA_1'] * test_df['CAR_TT']
    + betas['B_time_income_02'] * test_df['INCOME_0'] * test_df['CAR_TT']
    + betas['B_time_income_12'] * test_df['INCOME_1'] * test_df['CAR_TT']
    + betas['B_time_income_22'] * test_df['INCOME_2'] * test_df['CAR_TT']
    + betas['B_time_who_12'] * test_df['WHO_1'] * test_df['CAR_TT']
    + betas['B_time_who_22'] * test_df['WHO_2'] * test_df['CAR_TT']
    + betas['B_time_purpose_12'] * test_df['PURPOSE_1'] * test_df['CAR_TT']
    + betas['B_time_purpose_22'] * test_df['PURPOSE_2'] * test_df['CAR_TT']
    + betas['B_time_purpose_32'] * test_df['PURPOSE_3'] * test_df['CAR_TT']
    + betas['B_time_male_2'] * test_df['MALE_1'] * test_df['CAR_TT']
    
    + betas['B_age_12'] * test_df['AGE_1'] 
    + betas['B_age_22'] * test_df['AGE_2'] 
    + betas['B_age_32'] * test_df['AGE_3'] 
    + betas['B_age_42'] * test_df['AGE_4'] 
    + betas['B_lugg_12'] * test_df['LUGGAGE_1']
    + betas['B_lugg_22'] * test_df['LUGGAGE_2'] 
    + betas['B_ga_2'] * test_df['GA_1'] 
    + betas['B_income_02'] * test_df['INCOME_0'] 
    + betas['B_income_12'] * test_df['INCOME_1'] 
    + betas['B_income_22'] * test_df['INCOME_2'] 
    + betas['B_who_12'] * test_df['WHO_1'] 
    + betas['B_who_22'] * test_df['WHO_2'] 
    + betas['B_purpose_12'] * test_df['PURPOSE_1'] 
    + betas['B_purpose_22'] * test_df['PURPOSE_2'] 
    + betas['B_purpose_32'] * test_df['PURPOSE_3'] 
    + betas['B_male_2'] * test_df['MALE_1'] 
    
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

# ============= Estimated beta parameters ==========
#βTT-TRAIN
estimated_beta_TT_TRAIN = (
    betas['B_time0_0']
    + betas['B_time_age_10'] * test_df['AGE_1']
    + betas['B_time_age_20'] * test_df['AGE_2']
    + betas['B_time_age_30'] * test_df['AGE_3']
    + betas['B_time_age_40'] * test_df['AGE_4']
    + betas['B_time_lugg_10'] * test_df['LUGGAGE_1']
    + betas['B_time_lugg_20'] * test_df['LUGGAGE_2']
    + betas['B_time_ga_0'] * test_df['GA_1']
    + betas['B_time_income_00'] * test_df['INCOME_0']
    + betas['B_time_income_10'] * test_df['INCOME_1']
    + betas['B_time_income_20'] * test_df['INCOME_2']
    + betas['B_time_who_10'] * test_df['WHO_1']
    + betas['B_time_who_20'] * test_df['WHO_2']
    + betas['B_time_purpose_10'] * test_df['PURPOSE_1']
    + betas['B_time_purpose_20'] * test_df['PURPOSE_2']
    + betas['B_time_purpose_30'] * test_df['PURPOSE_3']
    + betas['B_time_male_0'] * test_df['MALE_1']
)

#βHE-TRAIN
estimated_beta_HE_TRAIN = (
    betas['B_headway0_0']
    + betas['B_headway_age_10'] * test_df['AGE_1']
    + betas['B_headway_age_20'] * test_df['AGE_2']
    + betas['B_headway_age_30'] * test_df['AGE_3']
    + betas['B_headway_age_40'] * test_df['AGE_4']
    + betas['B_headway_lugg_10'] * test_df['LUGGAGE_1']
    + betas['B_headway_lugg_20'] * test_df['LUGGAGE_2']
    + betas['B_headway_ga_0'] * test_df['GA_1']
    + betas['B_headway_income_00'] * test_df['INCOME_0']
    + betas['B_headway_income_10'] * test_df['INCOME_1']
    + betas['B_headway_income_20'] * test_df['INCOME_2']
    + betas['B_headway_who_10'] * test_df['WHO_1']
    + betas['B_headway_who_20'] * test_df['WHO_2']
    + betas['B_headway_purpose_10'] * test_df['PURPOSE_1']
    + betas['B_headway_purpose_20'] * test_df['PURPOSE_2']
    + betas['B_headway_purpose_30'] * test_df['PURPOSE_3']
    + betas['B_headway_male_0'] * test_df['MALE_1']
)

#βTT-SM
estimated_beta_TT_SM = (
    betas['B_time0_1']
    + betas['B_time_age_11'] * test_df['AGE_1']
    + betas['B_time_age_21'] * test_df['AGE_2']
    + betas['B_time_age_31'] * test_df['AGE_3']
    + betas['B_time_age_41'] * test_df['AGE_4']
    + betas['B_time_lugg_11'] * test_df['LUGGAGE_1']
    + betas['B_time_lugg_21'] * test_df['LUGGAGE_2']
    + betas['B_time_ga_1'] * test_df['GA_1']
    + betas['B_time_income_01'] * test_df['INCOME_0']
    + betas['B_time_income_11'] * test_df['INCOME_1']
    + betas['B_time_income_21'] * test_df['INCOME_2']
    + betas['B_time_who_11'] * test_df['WHO_1']
    + betas['B_time_who_21'] * test_df['WHO_2']
    + betas['B_time_purpose_11'] * test_df['PURPOSE_1']
    + betas['B_time_purpose_21'] * test_df['PURPOSE_2']
    + betas['B_time_purpose_31'] * test_df['PURPOSE_3']
    + betas['B_time_male_1'] * test_df['MALE_1']
)

#βHE-SM
estimated_beta_HE_SM = (
    betas['B_headway0_1']
    + betas['B_headway_age_11'] * test_df['AGE_1']
    + betas['B_headway_age_21'] * test_df['AGE_2']
    + betas['B_headway_age_31'] * test_df['AGE_3']
    + betas['B_headway_age_41'] * test_df['AGE_4']
    + betas['B_headway_lugg_11'] * test_df['LUGGAGE_1']
    + betas['B_headway_lugg_21'] * test_df['LUGGAGE_2']
    + betas['B_headway_ga_1'] * test_df['GA_1']
    + betas['B_headway_income_01'] * test_df['INCOME_0']
    + betas['B_headway_income_11'] * test_df['INCOME_1']
    + betas['B_headway_income_21'] * test_df['INCOME_2']
    + betas['B_headway_who_11'] * test_df['WHO_1']
    + betas['B_headway_who_21'] * test_df['WHO_2']
    + betas['B_headway_purpose_11'] * test_df['PURPOSE_1']
    + betas['B_headway_purpose_21'] * test_df['PURPOSE_2']
    + betas['B_headway_purpose_31'] * test_df['PURPOSE_3']
    + betas['B_headway_male_1'] * test_df['MALE_1']
)

#βSEATS-SM
estimated_beta_SEATS_SM = (
    betas['B_seats0_1']
    + betas['B_seats_age_11'] * test_df['AGE_1']
    + betas['B_seats_age_21'] * test_df['AGE_2']
    + betas['B_seats_age_31'] * test_df['AGE_3']
    + betas['B_seats_age_41'] * test_df['AGE_4']
    + betas['B_seats_lugg_11'] * test_df['LUGGAGE_1']
    + betas['B_seats_lugg_21'] * test_df['LUGGAGE_2']
    + betas['B_seats_ga_1'] * test_df['GA_1']
    + betas['B_seats_income_01'] * test_df['INCOME_0']
    + betas['B_seats_income_11'] * test_df['INCOME_1']
    + betas['B_seats_income_21'] * test_df['INCOME_2']
    + betas['B_seats_who_11'] * test_df['WHO_1']
    + betas['B_seats_who_21'] * test_df['WHO_2']
    + betas['B_seats_purpose_11'] * test_df['PURPOSE_1']
    + betas['B_seats_purpose_21'] * test_df['PURPOSE_2']
    + betas['B_seats_purpose_31'] * test_df['PURPOSE_3']
    + betas['B_seats_male_1'] * test_df['MALE_1']
)

#βTT-CAR
estimated_beta_TT_CAR = (
    betas['B_time0_2']
    + betas['B_time_age_12'] * test_df['AGE_1']
    + betas['B_time_age_22'] * test_df['AGE_2']
    + betas['B_time_age_32'] * test_df['AGE_3']
    + betas['B_time_age_42'] * test_df['AGE_4']
    + betas['B_time_lugg_12'] * test_df['LUGGAGE_1']
    + betas['B_time_lugg_22'] * test_df['LUGGAGE_2']
    + betas['B_time_ga_2'] * test_df['GA_1']
    + betas['B_time_income_02'] * test_df['INCOME_0']
    + betas['B_time_income_12'] * test_df['INCOME_1']
    + betas['B_time_income_22'] * test_df['INCOME_2']
    + betas['B_time_who_12'] * test_df['WHO_1']
    + betas['B_time_who_22'] * test_df['WHO_2']
    + betas['B_time_purpose_12'] * test_df['PURPOSE_1']
    + betas['B_time_purpose_22'] * test_df['PURPOSE_2']
    + betas['B_time_purpose_32'] * test_df['PURPOSE_3']
    + betas['B_time_male_2'] * test_df['MALE_1']
)

estimated_ASC_TRAIN = betas['ASC_0']
estimated_ASC_SM = (
    betas['ASC_1']
    + betas['B_age_11'] * test_df['AGE_1']
    + betas['B_age_21'] * test_df['AGE_2']
    + betas['B_age_31'] * test_df['AGE_3']
    + betas['B_age_41'] * test_df['AGE_4']
    + betas['B_lugg_11'] * test_df['LUGGAGE_1']
    + betas['B_lugg_21'] * test_df['LUGGAGE_2']
    + betas['B_ga_1'] * test_df['GA_1']
    + betas['B_income_01'] * test_df['INCOME_0']
    + betas['B_income_11'] * test_df['INCOME_1']
    + betas['B_income_21'] * test_df['INCOME_2']
    + betas['B_who_11'] * test_df['WHO_1']
    + betas['B_who_21'] * test_df['WHO_2']
    + betas['B_purpose_11'] * test_df['PURPOSE_1']
    + betas['B_purpose_21'] * test_df['PURPOSE_2']
    + betas['B_purpose_31'] * test_df['PURPOSE_3']
    + betas['B_male_1'] * test_df['MALE_1']
)


estimated_beta_TT_TRAIN_mean = estimated_beta_TT_TRAIN.mean()
estimated_beta_HE_TRAIN_mean = estimated_beta_HE_TRAIN.mean()
estimated_beta_TT_SM_mean = estimated_beta_TT_SM.mean()
estimated_beta_HE_SM_mean = estimated_beta_HE_SM.mean()
estimated_beta_SEATS_SM_mean = estimated_beta_SEATS_SM.mean()
estimated_beta_TT_CAR_mean = estimated_beta_TT_CAR.mean()

estimated_beta_TT_TRAIN_std = estimated_beta_TT_TRAIN.std()
estimated_beta_HE_TRAIN_std = estimated_beta_HE_TRAIN.std()
estimated_beta_TT_SM_std = estimated_beta_TT_SM.std()
estimated_beta_HE_SM_std = estimated_beta_HE_SM.std()
estimated_beta_SEATS_SM_std = estimated_beta_SEATS_SM.std()
estimated_beta_TT_CAR_std = estimated_beta_TT_CAR.std()

print("==== mean estimated beta values ====")
print("b_TT-TRAIN:", estimated_beta_TT_TRAIN_mean)
print("b_HE-TRAIN:", estimated_beta_HE_TRAIN_mean)
print("b_TT-SM:", beta_TT_SM_mean)
print("b_HE-SM:", beta_HE_SM_mean)
print("b_SEATS-SM:", beta_SEATS_SM_mean)
print("b_TT-CAR:", beta_TT_CAR_mean)

print("==== std estimated beta values ====")
print("b_TT-TRAIN:", estimated_beta_TT_TRAIN_std)
print("b_HE-TRAIN:", estimated_beta_HE_TRAIN_std)
print("b_TT-SM:", estimated_beta_TT_SM_std)
print("b_HE-SM:", estimated_beta_HE_SM_std)
print("b_SEATS-SM:", estimated_beta_SEATS_SM_std)
print("b_TT-CAR:", estimated_beta_TT_CAR_std)
print("==== estimated asc values ====")
print("ASC_TRAIN:", estimated_ASC_TRAIN)
print("ASC_SM:", estimated_ASC_SM.mean())

# ------------ Plots ---------------
mask = ((test_df["INCOME_0"] == 1) | (test_df["INCOME_1"] == 1) | (test_df["INCOME_2"] == 1 ))
X_masked = test_df[mask].copy()
income_cols = ['INCOME_0', 'INCOME_1', 'INCOME_2']

# A single categorical column
X_masked['INCOME_GROUP'] = (
X_masked[income_cols]
.idxmax(axis=1)
.str.replace('INCOME_', '', regex=False)
)

X_masked['pred_vots'] = -estimated_beta_TT_TRAIN
X_masked['pred_vohes'] = -estimated_beta_HE_TRAIN
X_masked.to_pickle("vot_vohe_testdata.pkl")