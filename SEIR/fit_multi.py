import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from posterior_plot import plot_posterior_kde_grid
from cmdstanpy import CmdStanModel, cmdstan_path
import arviz as az
from pathlib import Path
from datetime import datetime

# Load data
df = pd.read_csv('Data/synthetic_dataset_no_coupling.csv')

N = np.array([11.5e6, 4.5e6, 6.8e6])  # Population sizes
# N = np.array([1100000, 440000, 680000])

# beta = np.array([0.3, 0.4, 0.35])*7  # Transmission rate per week (R0 = beta/gamma)
# sigma = (1/8.5)*7 #incubation rate per week
# gamma = (1/10)*7   # Recovery rate per week

beta = np.array([0.3, 0.4, 0.35])*7  # Transmission rate (R0 = beta/gamma)
sigma = np.array([1/8.5, 1/8.5, 1/8.5])*7
gamma = np.array([1/13, 1/14, 1/13])*7
phi = 10           # Negative binomial dispersion parameter
reporting_rate = 1

# Define initial conditions
g0 = [N[0]-20, 10, 10, 0, 0]
l0 = [N[1]-20, 10, 10, 0, 0]
s0 = [N[2]-20, 10, 10, 0, 0] 
y0 = [g0, l0, s0]

# Generate timestamp for uniqueness
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # e.g., 20250607_103939

# Extract cases for the current country (assuming column names like 'Guinea_Noise', etc.)
weekly_cases = df[['Guinea_Noise', 'Liberia_Noise', 'SierraLeone_Noise']]
cases = weekly_cases.values.T

# Make the data structure
seir_data = {
    'N_countries': 3,
    "n_weeks": len(cases[0]),
    "y0": y0,
    "t0": 0,
    "t": np.arange(1, len(cases[0]) + 1),
    "N": N.astype(int),
    "cases": cases,
    'beta_values': beta,
    'sigma_values': sigma,
    'gamma_values': gamma,
    "reporting_rate": reporting_rate
}

# Fit the model
print(f'Running model fit for All Patches at time {timestamp}...')
model = CmdStanModel(stan_file = 'stan files/mult.stan')
print(f'\nDone compiling...Now Sampling!')

fit = model.sample(data=seir_data,
                    iter_sampling=2000,
                    chains=4,
                    seed=2)

# Save the fit output with a memorable name
output_dir = f'outputs/all_Patches_{timestamp}'
Path(output_dir).mkdir(parents=True, exist_ok=True)

fit.save_csvfiles(dir=output_dir)
print(f"\nFit for all Patches saved to: {output_dir}")

if __name__ == "__main__":
    print("Fits for all Patches completed. Check 'output/' directory.")