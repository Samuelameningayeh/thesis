functions {
  vector seir(real t,
             vector y,
             int n_patches,
             array[] real k_j,
             array[,] real beta_kj,
             real sigma,
             real gamma,
             array[] int N,
             array[,] real r_ij,
             array[,] real g_m_ji) {

      vector[4 * n_patches * n_patches + n_patches * n_patches] dydt; // S_ij, E_ij, I_ij, R_ij for each i,j, plus incidence
      array[n_patches] real N_j; // Total population in patch j

      // Compute total population in each patch j
      for (j in 1:n_patches) {
        N_j[j] = 0;
        for (i in 1:n_patches) {
          N_j[j] += y[(i-1)*4*n_patches + (j-1)*4 + 1] + // S_ij
                    y[(i-1)*4*n_patches + (j-1)*4 + 2] + // E_ij
                    y[(i-1)*4*n_patches + (j-1)*4 + 3] + // I_ij
                    y[(i-1)*4*n_patches + (j-1)*4 + 4];  // R_ij
        }
      }

      // Dynamics for each resident patch i, currently in patch j
      for (i in 1:n_patches) {
        for (j in 1:n_patches) {
          real S_ij = y[(i-1)*4*n_patches + (j-1)*4 + 1];
          real E_ij = y[(i-1)*4*n_patches + (j-1)*4 + 2];
          real I_ij = y[(i-1)*4*n_patches + (j-1)*4 + 3];
          real R_ij = y[(i-1)*4*n_patches + (j-1)*4 + 4];

          real S_ii = y[(i-1)*4*n_patches + (i-1)*4 + 1];
          real E_ii = y[(i-1)*4*n_patches + (i-1)*4 + 2];
          real I_ii = y[(i-1)*4*n_patches + (i-1)*4 + 3];
          real R_ii = y[(i-1)*4*n_patches + (i-1)*4 + 4];

          // Transmission term: sum_k (k_j * beta_kj * I_kj / N_j)
          real transmission = 0;
          for (k in 1:n_patches) {
            real I_kj = y[(k-1)*4*n_patches + (j-1)*4 + 3];
            transmission += k_j[j] * beta_kj[k, j] * I_kj / N_j[j];
          }

          // S_ij dynamics
          dydt[(i-1)*4*n_patches + (j-1)*4 + 1] = -S_ij * transmission + g_m_ji[j, i] * S_ii - r_ij[i, j] * S_ij;

          // E_ij dynamics
          dydt[(i-1)*4*n_patches + (j-1)*4 + 2] = S_ij * transmission - sigma * E_ij + g_m_ji[j, i] * E_ii - r_ij[i, j] * E_ij;

          // I_ij dynamics
          dydt[(i-1)*4*n_patches + (j-1)*4 + 3] = sigma * E_ij - gamma * I_ij + g_m_ji[j, i] * I_ii - r_ij[i, j] * I_ij;

          // R_ij dynamics
          dydt[(i-1)*4*n_patches + (j-1)*4 + 4] = gamma * I_ij + g_m_ji[j, i] * R_ii - r_ij[i, j] * R_ij;

          // Incidence (new infections for observation)
          dydt[4*n_patches*n_patches + (i-1)*n_patches + j] = sigma * E_ij;
        }
      }

      return dydt;
  }
}
data {
  int<lower=1> n_patches;           // Number of patches
  int<lower=1> n_days;              // Number of days
  array[n_patches, n_patches] vector[5] y0; // Initial conditions for each (i,j) pair
  real t0;                          // Initial time
  array[n_days] real t;             // Time points
  array[n_patches] int N;           // Total population of residents in patch i
  array[n_patches, n_days] int<lower=0> cases; // Observed cases per patch i
  array[n_patches, n_patches] real<lower=0> r_ij; // Movement rates for residents of i from j to elsewhere
  array[n_patches, n_patches] real<lower=0> g_m_ji; // Return rates (g_i * m_ji)
  array[n_patches] real<lower=0> k_j; // Contact rate in patch j
  array[N_countries] real beta_values; 
  array[N_countries] real sigma_values;
  array[N_countries] real gamma_values; 
  real<lower=0, upper=1> reporting_rate;
}
parameters {
  real<lower=0> sigma;              // Progression rate
  array[n_patches, n_patches] real<lower=0> beta_kj; // Transmission probability for residents of k in patch j
  real<lower=0> gamma;              // Recovery rate
  real<lower=0> phi_inv;            // Dispersion parameter
}
transformed parameters {
  array[N_countries, n_weeks] real adjusted_incidence;
  array[n_patches, n_patches, n_days] vector[5] y; // Solutions: S_ij, E_ij, I_ij, R_ij, incidence
  array[n_patches, n_days-1*n_patches, n_days] incidence;       // Incidence for residents of patch i
  real<lower=0> phi = 1. / phi_inv;
  //real<lower=0> reporting_rate = 0.8;

  // Flatten y0 for ODE solver
  vector[4*n_patches*n_patches + n_patches*n_patches] y0_flat;
  for (i in 1:n_patches) {
    for (j in 1:n_patches) {
      y0_flat[(i-1)*4*n_patches + (j-1)*4 + 1] = y0[i, j, 1]; // S_ij
      y0_flat[(i-1)*4*n_patches + (j-1)*4 + 2] = y0[i, j, 2]; // E_ij
      y0_flat[(i-1)*4*n_patches + (j-1)*4 + 3] = y0[i, j, 3]; // I_ij
      y0_flat[(i-1)*4*n_patches + (j-1)*4 + 4] = y0[i, j, 4]; // R_ij
      y0_flat[4*n_patches*n_patches + (i-1)*n_patches + j] = 0; // Incidence
    }
  }

  // Solve ODE for each time point
  {
    array[n_days] vector[4*n_patches*n_patches + n_patches*n_patches] y_flat;
    y_flat = ode_rk45(seir, y0_flat, t0, t, n_patches, k_j, beta_kj, sigma, gamma, N, r_ij, g_m_ji);

    // Unflatten results
    for (i in 1:n_patches) {
      for (j in 1:n_patches) {
        for (d in 1:n_days) {
          y[i, j, d, 1] = y_flat[d, (i-1)*4*n_patches + (j-1)*4 + 1]; // S_ij
          y[i, j, d, 2] = y_flat[d, (i-1)*4*n_patches + (j-1)*4 + 2]; // E_ij
          y[i, j, d, 3] = y_flat[d, (i-1)*4*n_patches + (j-1)*4 + 3]; // I_ij
          y[i, j, d, 4] = y_flat[d, (i-1)*4*n_patches + (j-1)*4 + 4]; // R_ij
          y[i, j, d, 5] = y_flat[d, 4*n_patches*n_patches + (i-1)*n_patches + j]; // Incidence
        }
      }
    }
  }

  // Compute total incidence for residents of patch i (summing over all j)
  for (i in 1:n_patches) {
    for (d in 1:n_days) {
      incidence[i, d] = 0;
      for (j in 1:n_patches) {
        if (d == 1) {
          incidence[i, d] += y[i, j, d, 5];
        } else {
          incidence[i, d] += y[i, j, d, 5] - y[i, j, d-1, 5];
          adjusted_incidence[i, d] = reporting_rate * incidence[i, d]+0.000005;
        }
      }
    }
  }
}
model {
  // Priors
  for (k in 1:n_patches) {
    for (j in 1:n_patches) {
      beta_kj[k, j] ~ lognormal(log(0.35), 0.5); // Transmission rate prior
    }
  }
  sigma ~ lognormal(1.0 / 4, 0.5);       // Prior mean: 5-day incubation period
  gamma ~ lognormal(1.0 / 6, 0.5);       // Prior mean: 7-day infectious period
  phi_inv ~ exponential(2);

  // Sampling distribution
  for (i in 1:n_patches) {
    cases[i] ~ neg_binomial_2(adjusted_incidence[i], phi);
  }
}
generated quantities {
  real R0;  // Approximate R0 as a weighted average
  array[n_patches] real recovery_time = 1 / gamma;
  array[n_patches] real incubation_period = 1 / sigma;
  array[n_patches, n_days] real pred_incidence;

  // Approximate R0 (weighted by residence times, simplified)
  R0 = 0;
  for (j in 1:n_patches) {
    real sum_beta_kj = 0;
    for (k in 1:n_patches) {
      sum_beta_kj += k_j[j] * beta_kj[k, j];
    }
    R0 += (sum_beta_kj / sigma) * (N_j[j] / sum(N)); // Weight by population in patch j
  }

  for (i in 1:n_patches) {
    pred_incidence[i] = neg_binomial_2_rng(adjusted_incidence[i], phi);
  }
}