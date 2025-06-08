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

# Country-specific data and parameters
countries = ['Guinea', 'Liberia', 'SierraLeone']
N = np.array([11.5e6, 4.5e6, 6.8e6])  # Population sizes
# N = np.array([1100000, 440000, 680000])

beta = np.array([0.3, 0.4, 0.35])*7  # Transmission rate per week (R0 = beta/gamma)
sigma = 1/8.5*7 #incubation rate per week
gamma = 1/10*7   # Recovery rate per week
phi = 10           # Negative binomial dispersion parameter
reporting_rate = 1
i0 = [10, 10, 10]  # Initial infected
e0 = [10, 10, 10]    # Initial exposed
r0 = [0, 0, 0]     # Initial recovered
c0 = [0, 0, 0]     # Initial cumulative cases


for i, country in enumerate(countries):
    # Extract cases for the current country (assuming column names like 'Guinea_Noise', etc.)
    weekly_cases = df[f'{country}_Noise']
    cases = weekly_cases.values

    # Generate timestamp for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # e.g., 20250607_103939

    # Define initial conditions
    s0 = N[i] - i0[i] - e0[i]
    y0 = [s0, e0[i], i0[i], r0[i], c0[i]]

    # Make the data structure
    seir_data = {
        "n_days": len(cases),
        "y0": y0,
        "t0": 0,
        "t": np.arange(1, len(cases) + 1),
        "N": int(N[i]),
        "cases": cases,
        'beta_value': beta[i],
        'sigma_value': sigma,
        'gamma_value': gamma,
        "reporting_rate": reporting_rate
    }

    # Fit the model
    print(f'Compiling stan file for {country}')
    model = CmdStanModel(stan_file='stan files/simple_seir.stan')
    print(f'\nDone compiling\n')

    print(f'Running model fit for {country} at time {timestamp}...')
    fit = model.sample(data=seir_data,
                       iter_sampling=2000,
                       chains=4,
                       seed=2)

    # Save the fit output with a memorable name
    output_dir = f'outputs/{country.lower()}_{timestamp}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fit.save_csvfiles(dir=output_dir)
    print(f"Fit for {country} saved to: {output_dir}\n")

if __name__ == "__main__":
    print("Fits for all countries completed. Check 'output/' directory.")