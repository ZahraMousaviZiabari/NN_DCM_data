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

def csv_to_dict(dir, filename, dev_av = False, save = False):

    # Read CSV file
    df_train = pd.read_csv(dir + "train_" + filename)
    if (dev_av == True):
        df_dev = pd.read_csv(dir + "dev_" + filename)

    df_test = pd.read_csv(dir + "test_" + filename)

    params = pickle.load(open("toy_data/params.pkl", "rb"))

    # vots and vowts
    base_vars = ['INC', 'FULL', 'FLEX']
    interaction_pairs = [(0, 1), (0, 2), (1, 2)]
    true_coefs_time = [-0.1,-0.5,-0.1,0.05,-0.2,0.05,0.1]
    true_coefs_wait = [-0.2,-0.8,-0.3,0.1,-0.3,0.08,0.3]

    #Train
    Z_train = df_train[['INC', 'FULL', 'FLEX']].values
    N = len(df_train)
    D = 4
    X_train = np.zeros((N,D,2))
    X_train[:,1:,0] = df_train[['TRAIN_CO', 'TRAIN_TT', 'TRAIN_WT']].values
    X_train[:,1:,1] = df_train[['CAR_CO', 'CAR_TT', 'CAR_WT']].values
    X_train[:,0,1] = np.ones((N))
    y_train = df_train['CHOICE'].astype(int).values
    vots_train, vowts_train = value_of_x(
        df=df_train,
        coef_time=true_coefs_time,
        coef_wait=true_coefs_wait,
        base_vars=base_vars,
        interactions=interaction_pairs
            )

    #dev
    if (dev_av == True):
        Z_dev = df_dev[['INC', 'FULL', 'FLEX']].values
        N = len(df_dev)
        D = 4
        X_dev = np.zeros((N,D,2))
        X_dev[:,1:,0] = df_dev[['TRAIN_CO', 'TRAIN_TT', 'TRAIN_WT']].values
        X_dev[:,1:,1] = df_dev[['CAR_CO', 'CAR_TT', 'CAR_WT']].values
        X_dev[:,0,1] = np.ones((N))
        y_dev = df_dev['CHOICE'].astype(int).values
        vots_dev, vowts_dev = value_of_x(
            df=df_dev,
            coef_time=true_coefs_time,
            coef_wait=true_coefs_wait,
            base_vars=base_vars,
            interactions=interaction_pairs
                )

    #Test
    Z_test = df_test[['INC', 'FULL', 'FLEX']].values
    N = len(df_test)
    D = 4
    X_test = np.zeros((N,D,2))
    X_test[:,1:,0] = df_test[['TRAIN_CO', 'TRAIN_TT', 'TRAIN_WT']].values
    X_test[:,1:,1] = df_test[['CAR_CO', 'CAR_TT', 'CAR_WT']].values
    X_test[:,0,1] = np.ones((N))
    y_test= df_test['CHOICE'].astype(int).values
    vots_test, vowts_test = value_of_x(
        df=df_test,
        coef_time=true_coefs_time,
        coef_wait=true_coefs_wait,
        base_vars=base_vars,
        interactions=interaction_pairs
            )

    data_train = {"x": X_train, 
            "z": Z_train,
            "y": y_train,
            "vots": vots_train, 
            "vowts": vowts_train,
            "params": params,
            "nll": 0, 
            "acc": 0}
    if (dev_av == True):
        data_dev = {"x": X_dev, 
                "z": Z_dev,
                "y": y_dev,
                "vots": vots_dev, 
                "vowts": vowts_dev,
                "params": params,
                "nll": 0, 
                "acc": 0}
    data_test = {"x": X_test, 
            "z": Z_test,
            "y": y_test,
            "vots": vots_test, 
            "vowts": vowts_test,
            "params": params,
            "nll": 0, 
            "acc": 0}

    if (dev_av == True):    
        data = {
            "train": data_train,
            "dev": data_dev,
            "test": data_test
        }
    else:
        data = {
            "train": data_train,
            "test": data_test
        }
    if (save == True):
        # income
        plt.hist(Z_train[:,0]*60)
        plt.xlabel("Income ($ per hour)")
        plt.savefig("toy_data/hist_income.png", dpi=300)

        # Save as pickle
        df_train.to_pickle("toy_data/train_10k.pkl")
        if (dev_av == True):
            df_dev.to_pickle("toy_data/dev_10k.pkl")
        df_test.to_pickle("toy_data/test_10k.pkl")
        pd.to_pickle(data, "toy_data/data_10k.pkl")

    return data