functions {
  vector seird_coupled(real t,
                       vector y,
                       real[] beta,
                       real sigma,
                       real gamma,
                       real alpha,
                       real[] g,
                       real[] r,
                       real[,] m,
                       real[] N,
                       int n_countries) {
    vector[n_countries * 6 + n_countries * n_countries * 5] dydt; // Total states
    int idx = 1;

    // Extract states: S_i, E_i, I_i, R_i, D_i, CumInc_i for each country
    vector[n_countries] S;
    vector[n_countries] E;
    vector[n_countries] I;
    vector[n_countries] R;
    vector[n_countries] D;
    vector[n_countries] CumInc;
    for (c in 1:n_countries) {
      S[c] = y[idx]; idx += 1;
      E[c] = y[idx]; idx += 1;
      I[c] = y[idx]; idx += 1;
      R[c] = y[idx]; idx += 1;
      D[c] = y[idx]; idx += 1;
      CumInc[c] = y[idx]; idx += 1;
    }

    // Extract coupling states: S_ij, E_ij, I_ij, R_ij, D_ij for each i, j
    vector[n_countries] S_ij[n_countries, n_countries];
    vector[n_countries] E_ij[n_countries, n_countries];
    vector[n_countries] I_ij[n_countries, n_countries];
    vector[n_countries] R_ij[n_countries, n_countries];
    vector[n_countries] D_ij[n_countries, n_countries];
    
    for (i in 1:n_countries) {
      for (j in 1:n_countries) {
        S_ij[i, j] = y[idx]; idx += 1;
        E_ij[i, j] = y[idx]; idx += 1;
        I_ij[i, j] = y[idx]; idx += 1;
        R_ij[i, j] = y[idx]; idx += 1;
        D_ij[i, j] = y[idx]; idx += 1;
      }
    }

    // Compute total infected in each patch for transmission
    vector[n_countries] I_total;
    for (j in 1:n_countries) {
      I_total[j] = I[j];
      for (i in 1:n_countries) {
        I_total[j] += I_ij[i, j];
      }
    }

    // Dynamics for patch i
    for (i in 1:n_countries) {
      // Precompute coupling sums for each compartment
      real sum_rji_Sii = 0.0;
      real sum_rji_Eii = 0.0;
      real sum_rji_Iii = 0.0;
      real sum_rji_Rii = 0.0;
      for (j in 1:n_countries) {
        int r_idx = j * n_countries + i; // Index for r_ji
        sum_rji_Sii += r[r_idx] * S_ij[i, j];
        sum_rji_Eii += r[r_idx] * E_ij[i, j];
        sum_rji_Iii += r[r_idx] * I_ij[i, j];
        sum_rji_Rii += r[r_idx] * R_ij[i, j];
      }

      // Transmission term (same for S and E)
      real transmission = beta[i] * S[i] * I_total[i] / N[i];

      // Compute derivatives
      int base_idx = (i - 1) * 6 + 1; // Base index for this country in dydt
      dydt[base_idx]     = -transmission + sum_rji_Sii - g[i] * S[i]; // S_i'
      dydt[base_idx + 1] = transmission - sigma * E[i] + sum_rji_Eii - g[i] * E[i]; // E_i'
      dydt[base_idx + 2] = sigma * E[i] - (gamma + alpha) * I[i] + sum_rji_Iii - g[i] * I[i]; // I_i'
      dydt[base_idx + 3] = gamma * I[i] + sum_rji_Rii - g[i] * R[i]; // R_i'
      dydt[base_idx + 4] = alpha * I[i]; // D_i'
      dydt[base_idx + 5] = sigma * E[i]; // CumInc_i'
    }

    // Dynamics for patch i residents in patch j
    for (i in 1:n_countries) {
      for (j in 1:n_countries) {
        // Compute common terms for this i,j pair
        int r_idx = i * n_countries + j; // Index for r_ij
        real g_m = g[i] * m[i, j]; // g_i * m_ij
        real r_ij = r[r_idx]; // r_ij
        int base_idx = n_countries * 6 + ((i - 1) * n_countries + (j - 1)) * 5 + 1; // Base index in dydt

        // Transmission term for S_ij and E_ij
        real transmission_ij = beta[j] * S_ij[i, j] * I_total[j] / N[j];

        // Compute derivatives
        dydt[base_idx]     = -transmission_ij + g_m * S[i] - r_ij * S_ij[i, j]; // S_ij'
        dydt[base_idx + 1] = transmission_ij - sigma * E_ij[i, j] + g_m * E[i] - r_ij * E_ij[i, j]; // E_ij'
        dydt[base_idx + 2] = sigma * E_ij[i, j] - (gamma + alpha) * I_ij[i, j] + g_m * I[i] - r_ij * I_ij[i, j]; // I_ij'
        dydt[base_idx + 3] = gamma * I_ij[i, j] + g_m * R[i] - r_ij * R_ij[i, j]; // R_ij'
        dydt[base_idx + 4] = alpha * I_ij[i, j] + g_m * D[i]; // D_ij'
      }
    }

    return dydt;
  }
}

data {
  int<lower=1> N_countries;                // Number of countries (3)
  int<lower=1> n_days;                     // Number of time points
  array[N_countries] vector[6] y0;         // Initial conditions for each country (S, E, I, R, D, CumInc)
  array[N_countries, N_countries] vector[5] y0_ij; // Initial conditions for coupling (S_ij, E_ij, I_ij, R_ij, D_ij)
  real t0;                                 // Initial time
  array[n_days] real t;                    // Time points
  array[N_countries] int N;                // Population sizes
  array[N_countries, n_days] int<lower=0> cases; // Observed cases for each country
}

parameters {
  vector<lower=0>[N_countries] beta;       // Country-specific transmission rates
  real<lower=0> sigma;                     // Shared progression rate (E to I)
  real<lower=0> gamma;                     // Shared recovery rate
  real<lower=0> alpha;                     // Shared death rate
  vector<lower=0>[N_countries] g;          // Rate of leaving patch i
  vector<lower=0>[N_countries * N_countries] r; // Rate of return r_ij
  vector<lower=0,upper=1>[N_countries] reporting_rate; // Country-specific reporting rates
  real<lower=0> phi_inv;                   // Negative binomial overdispersion
}

transformed parameters {
  real phi = 1.0 / phi_inv;                // Negative binomial dispersion
  array[N_countries, n_days] real incidence; // Incidence for each country
  array[N_countries, n_days] vector[6] y;  // SEIRD states for each country
  array[N_countries, N_countries, n_days] vector[5] y_ij; // Coupling states

  // Construct full initial state vector
  vector[N_countries * 6 + N_countries * N_countries * 5] y0_full;
  int idx = 1;
  for (c in 1:N_countries) {
    for (k in 1:6) {
      y0_full[idx] = y0[c][k];
      idx += 1;
    }
  }
  for (i in 1:N_countries) {
    for (j in 1:N_countries) {
      for (k in 1:5) {
        y0_full[idx] = y0_ij[i, j][k];
        idx += 1;
      }
    }
  }

  // Solve ODE
  {
    real theta[3] = {sigma, gamma, alpha};
    array[N_countries] real beta_arr = to_array_1d(beta);
    array[N_countries] real g_arr = to_array_1d(g);
    array[N_countries * N_countries] real r_arr = to_array_1d(r);
    array[N_countries] real N_arr = to_array_1d(to_vector(N));
    real m[N_countries, N_countries]; // Movement matrix (fixed for simplicity)
    for (i in 1:N_countries) {
      for (j in 1:N_countries) {
        m[i, j] = i == j ? 0.0 : 0.1 / (N_countries - 1); // Uniform movement
      }
    }

    array[n_days] vector[N_countries * 6 + N_countries * N_countries * 5] y_full;
    y_full = ode_rk45(seird_coupled, y0_full, t0, t, beta_arr, sigma, gamma, alpha, g_arr, r_arr, m, N_arr, N_countries);

    // Extract states
    for (t_idx in 1:n_days) {
      idx = 1;
      for (c in 1:N_countries) {
        for (k in 1:6) {
          y[c, t_idx][k] = y_full[t_idx][idx];
          idx += 1;
        }
      }
      for (i in 1:N_countries) {
        for (j in 1:N_countries) {
          for (k in 1:5) {
            y_ij[i, j, t_idx][k] = y_full[t_idx][idx];
            idx += 1;
          }
        }
      }
    }
  }

  // Compute incidence
  for (c in 1:N_countries) {
    incidence[c, 1] = y[c, 1, 6]; // Initial incidence
    for (i in 2:n_days) {
      incidence[c, i] = y[c, i, 6] - y[c, i-1, 6];
    }
  }
}

model {
  // Priors
  beta ~ lognormal(log(0.35), 0.5);
  sigma ~ lognormal(log(1.0/4), 0.5);
  gamma ~ lognormal(log(1.0/6), 0.5);
  alpha ~ lognormal(log(0.01), 0.5); // Small death rate
  g ~ lognormal(log(0.1), 0.5); // Movement rate
  r ~ lognormal(log(0.2), 0.5); // Return rate
  reporting_rate ~ beta(8, 2); // Centered around 0.8
  phi_inv ~ exponential(2);

  // Likelihood
  for (c in 1:N_countries) {
    for (i in 1:n_days) {
      cases[c, i] ~ neg_binomial_2(max(reporting_rate[c] * incidence[c, i], 1e-10), phi);
    }
  }
}

generated quantities {
  vector[N_countries] R0 = beta / sigma;
  real recovery_time = 1.0 / gamma;
  real incubation_period = 1.0 / sigma;
  array[N_countries, n_days] real pred_incidence;

  for (c in 1:N_countries) {
    for (i in 1:n_days) {
      pred_incidence[c, i] = neg_binomial_2_rng(max(reporting_rate[c] * incidence[c, i], 1e-10), phi);
    }
  }
}