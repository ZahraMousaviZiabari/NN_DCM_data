# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 12:51:47 2025

@author: szmz
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:27:59 2025

@author: szmz
"""

import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def df_to_choice_tensor(df, alts, feats, asc_alt=["CAR"]):
    """
    Transform a wide-format dataframe into a (N, J, D) tensor,
    where:
        N = number of rows (individuals),
        J = number of alternatives,
        D = number of features.

    Parameters
    ----------
    df : pd.DataFrame
    alts : list of str
        List of alternative names.
    feats : list of str
        List of feature names.
    asc_alt : list of str
        Alternatives for which ASC = 1 (others = 0).

    Returns
    -------
    Z_new : np.ndarray
        Tensor of shape (N, J, D+1) with ASC appended.
    """

    # --- Sanity check ---
    if isinstance(asc_alt, str):
        asc_alt = [asc_alt]   # allow scalar input for convenience

    missing = set(asc_alt) - set(alts)
    if missing:
        raise ValueError(f"asc_alt contains unknown alternatives: {missing}")

    # Build full column names
    alt_feat_cols = [f"{a}_{f}" for a in alts for f in feats]

    # Subset only these columns
    df_sub = df[alt_feat_cols].copy()

    # Build MultiIndex (alt, feat)
    alt_feat = [c.split("_", 1) for c in alt_feat_cols]
    df_sub.columns = pd.MultiIndex.from_tuples(alt_feat, names=["alt", "feat"])

    # Ensure correct ordering
    df_sub = df_sub.reindex(columns=pd.MultiIndex.from_product([alts, feats]))

    # Reshape into (N, J, D)
    N, J, D = len(df_sub), len(alts), len(feats)
    Z = df_sub.to_numpy().reshape(N, J, D)

    # --- Build ASC vector ---
    asc = np.zeros((N, J, 1), dtype=int)

    # Set ASC = 1 for each selected alternative
    for a in asc_alt:
        j = alts.index(a)
        asc[:, j, 0] = 1

    # Concatenate ASC as the last feature
    Z_new = np.concatenate([Z, asc], axis=2)

    return Z_new


class MultinomialLogitChoice:
    def __init__(self, n_features, n_alternatives, lr=0.01, reg=0.0):
        self.n_features = n_features
        self.lr = lr
        self.reg = reg
        # Single coefficient vector (all alternatives share it)
        self.beta = 0.01 * np.random.randn(n_features)  # (F=D*K)
        self.asc = np.zeros(n_alternatives)
        self.asc[1:] = 0.01 * np.random.randn(n_alternatives-1) #(J)

    def _utilities(self, Z, X):
        """
        Compute U vectorized.
        Z: (N, J, D)
        X: (N, K)
        """
        N, J, D = Z.shape
        _, K = X.shape
    
        U = np.zeros((N, J))
    
        beta = self.beta
    
        for d in range(D):
            if d == D - 1:
                # ASC
                U += self.asc  # shape (J,)
                continue
    
            if d == D - 2:
                # Fixed coefficient = 1
                U -= Z[:, :, d]
                continue
    
            # ======================
            #   β BLOCK FOR THIS d
            # ======================
            base = d * (K + 1)
    
            # -------- main effect --------
            beta_main = beta[base]                        # scalar
            U += Z[:, :, d] * beta_main                   # (N,J)
    
            # -------- interactions --------
            # Prepare Z_d:        (N,J,1)
            Z_d = Z[:, :, d][:, :, None]
            # Prepare X:          (N,1,K)
            X_b = X[:, None, :]
            # Product:            (N,J,K)
            ZX = Z_d * X_b
    
            # β_inter for this d: shape (K,)
            beta_inter = beta[base + 1: base + 1 + K]
    
            # Contract over k-dimension
            U += ZX @ beta_inter   # (N,J)
        
        return U

        #return Z @ self.beta.T  # (N,J,D) @ (D,) = (N,J)

    def _softmax(self, U):
        """Softmax over alternatives j for each n."""
        U_shift = U - np.max(U, axis=1, keepdims=True)
        expU = np.exp(U_shift)
        return expU / np.sum(expU, axis=1, keepdims=True)

    def predict_proba(self, Z, X):
        return self._softmax(self._utilities(Z, X))

    def negative_log_likelihood(self, Z, X, y):
        """
        Z: (N,J,D), y: (N,) with chosen alternative indices
        """
        N, J, D = Z.shape
        nll = 0
        P = self.predict_proba(Z, X)
        y = y.flatten()
        chosen_probs = P[np.arange(N), y]
        nll -= np.sum(np.log(chosen_probs + 1e-12))
        if self.reg > 0:
            nll += 0.5 * self.reg * np.sum(self.beta ** 2)
        return nll

    def _gradients(self, Z, X, y):
        """
        Gradient wrt beta
        """
        N, J, D = Z.shape
        _, K = X.shape
        grad = np.zeros_like(self.beta)
        idx = 0  # beta index
        P = self.predict_proba(Z, X)  # (N,J)
        # indicator of chosen alternative
        Y = np.zeros_like(P)
        y = y.flatten()
        Y[np.arange(N), y] = 1
        # difference (P - Y): (N,J)
        diff = (P - Y)  # (N,J)
        # gradient: sum_n sum_j (p_{nj}-y_{nj}) x_{nj}
        
        for d in range(D):
            if d >= D - 2:
                #  no β
                grad_asc = np.sum(diff, axis=0)
            else:
                # -------- main effect beta index --------
                # ∂U/∂beta_main = Z[:,:,d]
                grad[idx] = np.sum(diff * Z[:,:,d])
                idx += 1

                # -------- interaction βs --------
                # ∂U/∂β_inter[k] = Z[:,:,d] * X[:,k]
                for k in range(K):
                    term = Z[:,:,d] * X[:,k][:,None]
                    grad[idx] = np.sum(diff * term)
                    idx += 1
            #grad[1:,D-1] = 0
        if self.reg > 0:
            grad += self.reg * self.beta
        return grad, grad_asc
    
    
    def fit(self, Z, X, y, inner_epochs=5, batch_size=None, optimizer="sgd",
        beta1=0.9, beta2=0.999, eps=1e-8, verbose=False):
        """
        Z: (N,J,D), y: (N,)
        optimizer: "sgd" or "adam"
        """
        N = Z.shape[0]
        indices = np.arange(N)
    
        # Adam states for beta
        m_beta = np.zeros_like(self.beta)
        v_beta = np.zeros_like(self.beta)
    
        # Adam states for ASC
        m_asc = np.zeros_like(self.asc[1:])
        v_asc = np.zeros_like(self.asc[1:])
    
        t = 0
    
        for epoch in range(1, inner_epochs + 1):
            np.random.shuffle(indices)
            Z_sh, X_sh, y_sh = Z[indices], X[indices], y[indices]
    
            # ---------------- Full-batch case ----------------
            if batch_size is None:
                grad, grad_asc = self._gradients(Z_sh, X_sh, y_sh)
    
                if optimizer == "sgd":
                    self.beta -= self.lr * grad
                    self.asc[1:]  -= self.lr * grad_asc[1:]
    
                elif optimizer == "adam":
                    t += 1
    
                    # ---- beta updates ----
                    m_beta = beta1 * m_beta + (1 - beta1) * grad
                    v_beta = beta2 * v_beta + (1 - beta2) * (grad ** 2)
                    m_hat = m_beta / (1 - beta1**t)
                    v_hat = v_beta / (1 - beta2**t)
                    self.beta -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
    
                    # ---- asc updates ----
                    m_asc = beta1 * m_asc + (1 - beta1) * grad_asc[1:]
                    v_asc = beta2 * v_asc + (1 - beta2) * (grad_asc[1:] ** 2)
                    m_hat_asc = m_asc / (1 - beta1**t)
                    v_hat_asc = v_asc / (1 - beta2**t)
                    self.asc[1:] -= self.lr * m_hat_asc / (np.sqrt(v_hat_asc) + eps)
    
            # ---------------- Mini-batch case ----------------
            else:
                for i in range(0, N, batch_size):
                    Zb = Z_sh[i:i+batch_size]
                    Xb = X_sh[i:i+batch_size]
                    yb = y_sh[i:i+batch_size]
    
                    grad, grad_asc = self._gradients(Zb, Xb, yb)
    
                    if optimizer == "sgd":
                        self.beta -= self.lr * grad
                        self.asc[1:]  -= self.lr * grad_asc[1:]
    
                    elif optimizer == "adam":
                        t += 1
    
                        # ---- beta updates ----
                        m_beta = beta1 * m_beta + (1 - beta1) * grad
                        v_beta = beta2 * v_beta + (1 - beta2) * (grad ** 2)
                        m_hat = m_beta / (1 - beta1**t)
                        v_hat = v_beta / (1 - beta2**t)
                        self.beta -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
    
                        # ---- asc updates ----
                        m_asc = beta1 * m_asc + (1 - beta1) * grad_asc[1:]
                        v_asc = beta2 * v_asc + (1 - beta2) * (grad_asc[1:]** 2)
                        m_hat_asc = m_asc / (1 - beta1**t)
                        v_hat_asc = v_asc / (1 - beta2**t)
                        self.asc[1:] -= self.lr * m_hat_asc / (np.sqrt(v_hat_asc) + eps)
    
            nll = self.negative_log_likelihood(Z, X, y)
            if verbose and (epoch % 5 == 0):
                print(f"Inner Epoch {epoch} | NLL: {nll:.4f}")
    
        return nll
    
        
            
# -------------------------
# Main
#--------------------------

def read_data() -> pd.DataFrame:
    """Read the data from file"""
    module_dir = os.path.dirname(__file__)  # Get the directory of the current file
    data_file_path = os.path.join(module_dir, 'data', 'train_10k_biogeme.csv') 
    
    # Read the data
    df = pd.read_csv(data_file_path)
    return df

# Load full dataset
df = read_data()
#df = df.sample(10000)


  
# Split data into training (80%) and testing (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# -------------------------
# Train
# -------------------------


X_train = train_df[['INC', 'FULL', 'FLEX']].copy()


X_train = X_train.values


Z_train = df_to_choice_tensor(train_df, alts  = ["TRAIN", "CAR"], feats = ["TT", "WT", "CO"], asc_alt="CAR")

y_mode = train_df['CHOICE'].astype(int).values

Z_to_be_estimated = ["TT", "WT"]
X_to_be_estimated = ["INC", "FULL", 'FLEX']
n_parameters = len(Z_to_be_estimated)* len(X_to_be_estimated) + len(Z_to_be_estimated) #(D*K) + D , asc will be considered separately

logit_model = MultinomialLogitChoice(n_features=n_parameters, n_alternatives=Z_train.shape[1], lr=0.0008, reg=1e-4)
nll = logit_model.fit(Z_train, X_train, y_mode, inner_epochs=3000, batch_size=1024, optimizer="adam", verbose=True)

# -------------------------
# Test
# -------------------------

X_test = test_df[['INC', 'FULL', 'FLEX']].copy()


X_test = X_test.values


Z_test = df_to_choice_tensor(test_df, alts  = ["TRAIN","CAR"], feats = ["TT", "WT", "CO"], asc_alt="CAR")

estimated_MNL_betas = logit_model.beta
estimated_MNL_betas = np.append(estimated_MNL_betas,logit_model.asc[1:])

beta_names = []
for z in Z_to_be_estimated:
    beta_names.append(f"b_{z}")
    for x in X_to_be_estimated:
        beta_names.append(f"b_{z}_{x}")

beta_names += [f"ASC_{j}" for j in range(1, len(logit_model.asc))]
estimated_MNL_betas_df = pd.DataFrame([estimated_MNL_betas], columns=beta_names)
estimated_MNL_betas_df = estimated_MNL_betas_df.T


P = logit_model.predict_proba(Z_test,X_test)

predicted_choice = np.where(P[:,0] > P[:,1], 0, 1)
correct_prediction = (predicted_choice == test_df['CHOICE']).astype(int)

"""
majority_class = np.bincount(test_df['CHOICE']).argmax()
baseline = np.mean(test_df['CHOICE'] == majority_class)
print("Baseline accuracy:", baseline)
"""

accuracy = correct_prediction.mean() * 100
error_rate = 100 - accuracy

# Compute performance metrics
print(f"Negative Log Likelihood: {nll:.4f}")
print("Estimated MNL Betas:\n", estimated_MNL_betas_df)
print(f"Prediction Accuracy: {accuracy:.2f}%")
print(f"Prediction Error Rate: {error_rate:.2f}%")