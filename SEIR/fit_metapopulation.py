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
df = pd.read_csv('Data/coupling_synthetic_ebola_dataset_with_death.csv')

weekly_coupling = df[['Guinea_Noise', 'Liberia_Noise', 'SierraLeone_Noise']]

N = np.array([11.5e6, 4.5e6, 6.8e6]) 

g0 = np.array([N[0]-20, 10, 10, 0])
l0 = np.array([N[1]-11, 10, 1, 0])
s0 = np.array([N[2]-11, 10, 1, 0])

y0 = np.concatenate([g0, l0, s0]).flatten()

ts = weekly_coupling.values.T

beta = np.array([[0.3, 0.03, 0.02], 
           [0.03, 0.4, 0.05], 
           [0.02, 0.05, 0.35]])*7
# beta = np.array([0.3, 0.4, 0.35])*7
sigma = np.array([1/8.5, 1/8.5, 1/8.5])*7
gamma = np.array([1/9, 1/7, 1/6])*7

phi = 10           # Negative binomial dispersion parameter
# alpha = 0.05*7
reporting_rate = 1  

# Make the data struture
seir_data = {
    'N_patches': 3,
    "n_weeks": len(ts[0]),
    'y0': y0,
    't0': 0,
    "t": np.arange(1, len(ts[0])+1),
    "N": N.astype(int),
    "cases": ts,
    'beta_data': beta,
    'sigma_data': sigma,
    'gamma_data': gamma,
    'reporting_rate': reporting_rate
}

# Generate timestamp for uniqueness
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # e.g., 20250607_103939

# Fit the model
print(f'Running Metapopulation model at time {timestamp}...')
model = CmdStanModel(stan_file = 'stan files/metapopulation.stan')

print(f'Done compiling...Now Sampling!')

fit = model.sample(data=seir_data,
                    iter_sampling=1000,
                    chains=4,
                    seed=0)

# Save the fit output with a memorable name
output_dir = f'outputs/metapopulation_{timestamp}'
Path(output_dir).mkdir(parents=True, exist_ok=True)

fit.save_csvfiles(dir=output_dir)
print(f"Fit for metapopulation saved to: {output_dir}")

if __name__ == "__main__":
    print("Fits for metapopulation completed. Check 'output/' directory.")