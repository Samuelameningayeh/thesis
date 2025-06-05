// Custom SEIR function for meta-population
functions{
  vector seir(real t,
              vector y,
              array[] real beta,
              real sigma,
              real gamma,
              real alpha,
              real N) {
    vector[6] dydt;
    real S = y[1];
    real E = y[2];
    real I = y[3];
    real R = y[4];
    real D = y[5];
    real C = y[6];

    real foi = 0;
    for (j in 1:3) {
      foi += beta[j] * S * I / N;
    }
    dydt[1] = -foi;                       // dS/dt (weekly)
    dydt[2] = foi - sigma * E;            // dE/dt (weekly)
    dydt[3] = sigma * E - gamma * I - alpha * I;        // dI/dt (weekly)
    dydt[4] = gamma * I;                    // dR/dt (weekly)
    dydt[5] = alpha * I;
    dydt[6] = sigma * E;

    return dydt;
  }
}

data {
  int<lower=1> N_patches;                // Number of patches (3: Guinea, Liberia, Sierra Leone)
  int<lower=1> n_weeks;                  // Number of weeks
  array[N_patches] vector[6] y0;         // Initial conditions (S, E, I, R, D, C) for each patch
  real t0;                               // Initial time
  array[n_weeks] real t;                 // Time points (weekly: 0, 1, 2, ...)
  array[N_patches] int N;                // Population sizes
  array[N_patches, n_weeks] int<lower=0> cases; // Observed weekly cases for each patch
  array[N_patches, N_patches] real<lower=0> beta_values; // Daily transmission rates between patches
  array[N_patches] real sigma_values;    // Daily sigma values
  array[N_patches] real gamma_values;    // Daily gamma values
  array[N_patches] real alpha_values; 
  real<lower=0, upper=1> reporting_rate;
}

parameters {
  array[N_patches, N_patches] real <lower=0> beta; // Patch-specific transmission rates (weekly)
  array[N_patches] real<lower=0> sigma;      // Patch-specific progression rate (E to I, weekly)
  array[N_patches] real<lower=0> gamma;      // Patch-specific recovery rate (weekly)
  array[N_patches] real<lower=0> alpha;
  real<lower=0> phi_inv;                     // Negative binomial overdispersion
}

transformed parameters {
  array[N_patches, n_weeks] vector[6] y;     // Weekly SEIR states for each patch
  array[N_patches, n_weeks] real weekly_incidence; // Weekly incidence for each patch
  real<lower=0> phi = 1.0 / phi_inv;         // Negative binomial dispersion
  array[N_patches, n_weeks] real adjusted_incidence;

  // Solve ODE at weekly intervals with rescaled rates
  for (p in 1:N_patches) {
    y[p] = ode_rk45(seir, y0[p], t0, t, beta[p], sigma[p], gamma[p], alpha[p], N[p]);
    
    // Compute weekly incidence (based on new infections)
    weekly_incidence[p, 1] = y[p, 1, 6]; // Initial incidence from E to I
    adjusted_incidence[p, 1] = (reporting_rate * weekly_incidence[p, 1]) + 0.00005;

    for (w in 2:n_weeks) {
      weekly_incidence[p, w] = y[p, w, 6] - y[p, w-1, 6];
      adjusted_incidence[p, w] = (reporting_rate * weekly_incidence[p, w]) + 0.00005;
    }
  }
}

model {
  // PRIORS (rescaled to weekly rates)
  for (p in 1:N_patches) {
    for (j in 1:N_patches) {
      beta[p, j] ~ lognormal(log(beta_values[p, j]), 0.5); // Weekly beta = daily beta / 7
    }
    sigma[p] ~ lognormal(log(sigma_values[p]), 0.5);     // Weekly sigma = daily sigma / 7
    gamma[p] ~ lognormal(log(gamma_values[p]), 0.5);     // Weekly gamma = daily gamma / 7
    alpha[p] ~ lognormal(log(alpha_data[p]), 0.5);
  }
  phi_inv ~ exponential(2);

  // LIKELIHOOD
  for (p in 1:N_patches) {
    cases[p] ~ neg_binomial_2(adjusted_incidence[p], phi);
  }
}

generated quantities {
  array[N_patches] real R0;              // Patch-specific R0 (weekly)
  array[N_patches] real recovery_time;   // Recovery time (weeks)
  array[N_patches] real incubation_period; // Incubation period (weeks)
  array[N_patches, n_weeks] real pred_incidence; // Predicted weekly cases

  for (p in 1:N_patches) {
    R0[p] = sum(beta[p]) / (gamma[p] + alpha[p]);     // R0 based on weekly rates
    recovery_time[p] = 1.0 / gamma[p];   // Weeks
    incubation_period[p] = 1.0 / sigma[p]; // Weeks
  }

  for (p in 1:N_patches) {
    pred_incidence[p] = neg_binomial_2_rng(adjusted_incidence[p], phi);
  }
}