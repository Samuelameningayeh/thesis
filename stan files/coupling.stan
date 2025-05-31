functions {
  vector seir(real t,
              vector y,
              array[] real beta_ii,  // Within-patch transmission rates
              array[,] real beta_ij, // Between-patch transmission rates
              array[] real sigma,
              array[] real gamma,
              array[] real N,
              int N_countries) {
    vector[5 * N_countries] dydt;  // 5 compartments per patch (S, E, I, R, Cumulative Incidence)

    // Loop over each patch
    for (i in 1:N_countries) {
      real S_i = y[5 * (i-1) + 1];  // S_i
      real E_i = y[5 * (i-1) + 2];  // E_i
      real I_i = y[5 * (i-1) + 3];  // I_i
      real R_i = y[5 * (i-1) + 4];  // R_i

      // Compute the coupling term: sum over all patches j
      real coupling_term = 0.0;
      for (j in 1:N_countries) {
        real I_j = y[5 * (j-1) + 3];  // I_j from patch j
        coupling_term += beta_ij[i, j] * S_i * I_j / N[j];
      }

      // SEIR equations with coupling
      real dS_i_dt = -coupling_term;  // S_i' = -sum_j(beta_ij * S_i * I_j / N_j)
      real dE_i_dt = coupling_term - sigma[i] * E_i;  // E_i' = sum_j(beta_ij * S_i * I_j / N_j) - sigma_i * E_i
      real dI_i_dt = sigma[i] * E_i - gamma[i] * I_i;  // I_i' = sigma_i * E_i - gamma_i * I_i
      real dR_i_dt = gamma[i] * I_i;  // R_i' = gamma_i * I_i

      // Assign derivatives
      dydt[5 * (i-1) + 1] = dS_i_dt;
      dydt[5 * (i-1) + 2] = dE_i_dt;
      dydt[5 * (i-1) + 3] = dI_i_dt;
      dydt[5 * (i-1) + 4] = dR_i_dt;
      dydt[5 * (i-1) + 5] = sigma[i] * E_i;  // Cumulative incidence
    }

    return dydt;
  }
}

data {
  int<lower=1> N_countries;                // Number of patches (countries)
  int<lower=1> n_weeks;                    // Number of weeks
  array[N_countries] vector[5] y0;         // Initial conditions for each patch
  real t0;                                 // Initial time
  array[n_weeks] real t;                   // Time points (weekly: 0, 1, 2, ...)
  array[N_countries] int N;                // Population sizes
  array[N_countries, n_weeks] int<lower=0> cases; // Observed weekly cases for each patch
  array[N_countries] real beta_values;     // Daily within-patch beta values (prior means)
  array[N_countries, N_countries] real beta_ij_values; // Daily between-patch beta values (prior means)
  array[N_countries] real sigma_values;    // Daily sigma values
  array[N_countries] real gamma_values;    // Daily gamma values
  real<lower=0, upper=1> reporting_rate;
}

parameters {
  array[N_countries] real<lower=0> beta_ii;    // Within-patch transmission rates (weekly)
  array[N_countries, N_countries] real<lower=0> beta_ij; // Between-patch transmission rates (weekly)
  array[N_countries] real<lower=0> sigma;      // Progression rates (E to I, weekly)
  array[N_countries] real<lower=0> gamma;      // Recovery rates (weekly)
  real<lower=0> phi_inv;                       // Negative binomial overdispersion
}

transformed parameters {
  array[N_countries, n_weeks] vector[5] y;   // Weekly SEIR states for each patch
  array[N_countries, n_weeks] real weekly_incidence; // Weekly incidence for each patch
  real<lower=0> phi = 1.0 / phi_inv;         // Negative binomial dispersion
  array[N_countries, n_weeks] real adjusted_incidence;

  // Flatten y for ODE solver
  vector[5 * N_countries] y0_flat;
  for (c in 1:N_countries) {
    for (i in 1:5) {
      y0_flat[5 * (c-1) + i] = y0[c][i];
    }
  }

  // Solve ODE at weekly intervals with rescaled rates
  {
    array[N_countries, n_weeks] vector[5 * N_countries] y_flat;
    y_flat = ode_rk45(seir, y0_flat, t0, t, beta_ii, beta_ij, sigma, gamma, N, N_countries);

    // Reshape y_flat back to y
    for (c in 1:N_countries) {
      for (w in 1:n_weeks) {
        for (i in 1:5) {
          y[c, w, i] = y_flat[c, w, 5 * (c-1) + i];
        }
      }
    }
  }

  // Compute weekly incidence
  for (c in 1:N_countries) {
    weekly_incidence[c, 1] = y[c, 1, 5];  // Initial incidence
    adjusted_incidence[c, 1] = (reporting_rate * weekly_incidence[c, 1]) + 0.00005;

    for (w in 2:n_weeks) {
      weekly_incidence[c, w] = y[c, w, 5] - y[c, w-1, 5];
      adjusted_incidence[c, w] = (reporting_rate * weekly_incidence[c, w]) + 0.00005;
    }
  }
}

model {
  // PRIORS (rescaled to weekly rates)
  for (p in 1:N_countries) {
    beta_ii[p] ~ lognormal(log(beta_values[p]), 0.3);  // Within-patch beta
    sigma[p] ~ lognormal(log(sigma_values[p]), 0.3);  // Weekly sigma
    gamma[p] ~ lognormal(log(gamma_values[p]), 0.3);  // Weekly gamma
  }
  for (i in 1:N_countries) {
    for (j in 1:N_countries) {
      beta_ij[i, j] ~ lognormal(log(beta_ij_values[i, j]), 0.3);  // Between-patch beta
    }
  }
  phi_inv ~ exponential(2);

  // LIKELIHOOD
  for (p in 1:N_countries) {
    cases[p] ~ neg_binomial_2(adjusted_incidence[p], phi);
  }
}

generated quantities {
  array[N_countries] real R0;   // Patch-specific R0 (weekly, within-patch only)
  array[N_countries] real recovery_time;        // Recovery time (weeks)
  array[N_countries] real incubation_period;    // Incubation period (weeks)
  array[N_countries, n_weeks] real pred_incidence; // Predicted weekly cases

  for (c in 1:N_countries) {
    R0[c] = (beta_ii[c] / gamma[c]);  // R0 based on within-patch transmission
    recovery_time[c] = (1.0 / gamma[c]);        // Weeks
    incubation_period[c] = (1.0 / sigma[c]);    // Weeks
  }

  for (c in 1:N_countries) {
    pred_incidence[c] = neg_binomial_2_rng(adjusted_incidence[c], phi);
  }
}