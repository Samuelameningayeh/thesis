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

beta = np.array([0.25, 0.27, 0.26])*7  # Transmission rate per week (R0 = beta/gamma)
sigma = (1/5)*7 #incubation rate per week
gamma = (1/6.6)*7   # Recovery rate per week
phi = 10           # Negative binomial dispersion parameter
reporting_rate = 1

# Define initial conditions
g0 = [N[0]-10, 10, 10, 0, 0]
l0 = [N[1]-10, 10, 1, 0, 0]
s0 = [N[2]-10, 10, 1, 0, 0] 
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
print(f'Running model fit for at time {timestamp}...')
model = CmdStanModel(stan_file = 'stan files/mult.stan')
print(f'Done compiling...Now Sampling!')

fit = model.sample(data=seir_data,
                    iter_warmup=500,
                    iter_sampling=1000,
                    chains=4,
                    seed=0)

# Save the fit output with a memorable name
output_dir = f'outputs/all_Patches_{timestamp}'
Path(output_dir).mkdir(parents=True, exist_ok=True)

fit.save_csvfiles(dir=output_dir)
print(f"Fit for all Patches saved to: {output_dir}")

if __name__ == "__main__":
    print("Fits for all Patches completed. Check 'output/' directory.")