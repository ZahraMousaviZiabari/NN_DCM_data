import pandas as pd
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_pickle("MNL_first_swissmetrodata_vot_vohe.pkl")

with open('tasteNet_swissmetro_vot_vohe.pkl', 'rb') as f:
    data = pickle.load(f)

with open('test.pkl', 'rb') as f:
    test_data = pickle.load(f)
sociodemo_atts = test_data['z'] # N,K
x_test_names = test_data['z_names']
X_test = pd.DataFrame(sociodemo_atts, columns=x_test_names) 

b_data = data.cpu().numpy()
df2 = pd.DataFrame(-b_data, columns=['TRAIN_TT', 'SM_TT', 'CAR_TT', 'TRAIN_HE', 'SM_HE', 'SM_SEATS', 'TRAIN_ONE', 'SM_ONE'])
income_cols = ['INCOME_0', 'INCOME_1', 'INCOME_2', 'INCOME_3']

df2['INCOME_GROUP'] = (
    X_test[income_cols]
    .idxmax(axis=1)
    .str.replace('INCOME_', '', regex=False)
    )
df2['pred_vots'] = df2['TRAIN_TT'].to_numpy()
df2['pred_vohes'] = df2['TRAIN_HE'].to_numpy()

df3 = pd.read_pickle("spec1_swissmetro_vot_vohe.pkl")
df4 = pd.read_pickle("spec2_swissmetro_vot_vohe.pkl")
#df5 = pd.read_pickle("combined_swissmetro_vot_vohe.pkl")

fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(18, 8),
    sharey='row'
)

# Row 1: TT (Models 1–4)
models = [
    df1,
    df2,
    df3,
    df4,
#    df5,
]

for j, X in enumerate(models):
    beta_TT = [
    X.loc[X['INCOME_GROUP'] == g, 'pred_vots']
    for g in ['0', '1', '2']
    ]

    axes[0, j].boxplot(
        beta_TT,
        labels=['Inc 0', 'Inc 1', 'Inc 2']
    )
    #axes[0, j].set_title(f'Model {j+1}')
    axes[0, 0].set_title('MNL-W1')
    axes[0, 1].set_title('TasteNet-MNL')
    axes[0, 2].set_title('Multiplicative')
    axes[0, 3].set_title('Additive')
    #axes[0, 4].set_title('Combined')
    axes[0, j].axhline(0, color='black', linewidth=0.8)
    axes[0, j].grid(axis='y', linestyle='--', alpha=0.6)

axes[0, 0].set_ylabel(r'$\beta_{TT,\mathrm{TRAIN}}$')


for j, X in enumerate(models):
    beta_HE = [
    X.loc[X['INCOME_GROUP'] == g, 'pred_vohes']
    for g in ['0', '1', '2']
    ]

    axes[1, j].boxplot(
        beta_HE,
        labels=['Inc 0', 'Inc 1', 'Inc 2']
    )

    #axes[1, j].set_title(f'Model {j+1}')
    axes[1, 0].set_title('MNL-W1')
    axes[1, 1].set_title('TasteNet-MNL')
    axes[1, 2].set_title('Multiplicative')
    axes[1, 3].set_title('Additive')
    #axes[1, 4].set_title('Combined')
    axes[1, j].axhline(0, color='black', linewidth=0.8)
    axes[1, j].grid(axis='y', linestyle='--', alpha=0.6)

axes[1, 0].set_ylabel(r'$\beta_{HE,\mathrm{TRAIN}}$')

plt.tight_layout()
plt.show()
fig.savefig('vot_vohes_boxplots.png')