import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("train_10k_rho_0.6_biogeme.csv")   # replace with your CSV filename

# Select the column to plot
column_name = "INC"         # replace with your column name

# Draw histogram
plt.figure(figsize=(8, 5))
plt.hist(df[column_name].dropna()*60, bins=15, range=(5, 80))
plt.xlabel("income ($ per hour)")
plt.ylabel("counts")
plt.tight_layout()

# Show plot
plt.show()
