from collections import OrderedDict
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

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

def csv_to_dict(dir, filename, save = False):

    # Read CSV file
    df_train = pd.read_csv(dir + "/train_" + filename +".csv")
    df_test = pd.read_csv(dir + "/test_" + filename +".csv")

    z_levels = OrderedDict()
    levels = [('MALE',2), ('AGE',5), ('INCOME',4), ('FIRST',2), ('WHO',3), ("PURPOSE",4), ("LUGGAGE",3), ('GA',2)]
    for elem in levels:
        z_levels[elem[0]]=elem[1]

    z_list = [[var+"_"+str(i) for i in range(z_levels[var])] for var in z_levels]
    z_names = []
    for elem in z_list:
        z_names.extend(elem)

    z_necessary = ["MALE_1", "AGE_1", "AGE_2", "AGE_3", "AGE_4", \
               "INCOME_0", "INCOME_1", "INCOME_2", "FIRST_1", "WHO_1", "WHO_2", \
               "PURPOSE_1", "PURPOSE_2", "PURPOSE_3", "LUGGAGE_1", "LUGGAGE_2", "GA_1"]
    z_names = z_necessary
    # get z
    z_train = df_train[z_necessary].values
    z_test = df_test[z_necessary].values

    x_names = ['TRAIN_TT', 'TRAIN_HE', 'TRAIN_CO', 'SM_TT','SM_HE', 'SM_SEATS','SM_CO', 'CAR_TT', 'CAR_CO']
    x_train = df_train[x_names].values
    y_train = df_train['CHOICE'].values +1
    x_test = df_test[x_names].values
    y_test = df_test['CHOICE'].values +1

    car_av_train = df_train['CAR_AV'].values
    car_av_test = df_test['CAR_AV'].values

    data_dict_train = {"x_names": x_names, "x": x_train, "z_names": z_names, \
             "z": z_train, "z_levels": z_levels, "y":y_train,'car_av': car_av_train}
    data_dict_test = {"x_names": x_names, "x": x_test, "z_names": z_names, \
             "z": z_test, "z_levels": z_levels,  "y":y_test, 'car_av': car_av_test}

    if (save == True):
        # Save as pickle
        pickle.dump(data_dict_train, open(f"{dir}/train_{filename}.pkl","wb"))
        pickle.dump(data_dict_test, open(f"{dir}/test_{filename}.pkl","wb"))

    return data_dict_train, data_dict_test