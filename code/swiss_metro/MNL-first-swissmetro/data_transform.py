import pickle
import pandas as pd
import numpy as np

def pkl_to_dataframe(filename):

    data = pickle.load(open("data/"+filename+".pkl", "rb"))
        
    service_atts = data["x"]# N,D
    z_names = data["x_names"]
    N = len(service_atts)
    
    Z = pd.DataFrame(service_atts, columns=z_names) 
    
    y = pd.DataFrame(data["y"], columns=["CHOICE"]) -1# N
    
    # Availability 
    av = np.ones((N, 3))
    av[:,2] = data["car_av"] # (N,3) av for all modes 
    
    x_all_names = data['z_names']
    x_levels = data['z_levels']
    sociodemo_atts = data['z'] # N,K
    X_all = pd.DataFrame(sociodemo_atts, columns=x_all_names) 

    # select x
    x_names = ["MALE_1", "AGE_1", "AGE_2", "AGE_3", "AGE_4", \
           "INCOME_0", "INCOME_1", "INCOME_2", "INCOME_3", "FIRST_1", "WHO_1", "WHO_2", \
           "PURPOSE_0", "PURPOSE_1", "PURPOSE_2", "PURPOSE_3", "LUGGAGE_1", "LUGGAGE_2", "GA_1"]
    X = X_all[x_names]
        
    return X, Z, y, x_names, av
