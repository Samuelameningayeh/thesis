functions {
  vector seir_metapop(
    real t,
    vector y,                         // [S1,E1,I1,R1, S2,E2,I2,R2, S3,E3,I3,R3]
    matrix beta_mat,                  // N_patches x N_patches
    vector sigma,                     // N_patches
    vector gamma,                     // N_patches
    vector N                          // N_patches
  ) {
    int n_patches = rows(beta_mat);
    vector[4 * n_patches] dydt;
    for (p in 1:n_patches) {
      real S_p = y[4*(p-1)+1];
      real E_p = y[4*(p-1)+2];
      real I_p = y[4*(p-1)+3];
      real R_p = y[4*(p-1)+4];
      // Coupled force of infection:
      real inf_sum = 0;
      for (j in 1:n_patches) {
        real I_j = y[4*(j-1)+3];
        inf_sum += beta_mat[p, j] * S_p * I_j / N[j];
      }
      dydt[4*(p-1)+1] = -inf_sum;
      dydt[4*(p-1)+2] = inf_sum - sigma[p] * E_p;
      dydt[4*(p-1)+3] = sigma[p] * E_p - gamma[p] * I_p;
      dydt[4*(p-1)+4] = gamma[p] * I_p;
    }
    return dydt;
  }
}


data {
  int<lower=1> N_patches;
  int<lower=1> n_weeks;
  vector[4*N_patches] y0;
  real t0;
  array[n_weeks] real t;
  vector[N_patches] N;
  array[N_patches, n_weeks] int<lower=0> cases;
  matrix[N_patches, N_patches] beta_data;   // prior means for β
  vector[N_patches] sigma_data;
  vector[N_patches] gamma_data;
  real<lower=0, upper=1> reporting_rate;
}
parameters {
  matrix<lower=0>[N_patches, N_patches] beta_mat;
  vector<lower=0>[N_patches] sigma;
  vector<lower=0>[N_patches] gamma;
  real<lower=0> phi_inv;
}

transformed parameters {
  array[n_weeks] vector[4*N_patches] y;
  array[N_patches, n_weeks] real weekly_incidence;
  array[N_patches, n_weeks] real adjusted_incidence;
  real<lower=0> phi = 1.0 / phi_inv;

  y = ode_rk45(seir_metapop, y0, t0, t, beta_mat, sigma, gamma, N);

  for (p in 1:N_patches) {
    weekly_incidence[p, 1] = sigma[p] * y[1, 4*(p-1)+2];
    adjusted_incidence[p, 1] = reporting_rate * weekly_incidence[p, 1];
    for (w in 2:n_weeks) {
      weekly_incidence[p, w] = sigma[p] * y[w-1, 4*(p-1)+2];
      adjusted_incidence[p, w] = reporting_rate * weekly_incidence[p, w];
    }
  }
}

model {
  // Priors centered on data
  for (p in 1:N_patches){
    for (j in 1:N_patches){
      beta_mat[p, j] ~ lognormal(log(beta_data[p, j]), 0.5);
    }
  }
  sigma ~ lognormal(log(sigma_data), 0.5);
  gamma ~ lognormal(log(gamma_data), 0.5);
  phi_inv ~ exponential(5);

  // Likelihood
  for (p in 1:N_patches)
    cases[p] ~ neg_binomial_2(adjusted_incidence[p], phi);
}

generated quantities {
  matrix[N_patches, N_patches] NGM;              // Next Generation Matrix
  array[N_patches, n_weeks] real pred_incidence;  // Posterior predictive cases
  vector[N_patches] incubation_period;            // Posterior incubation period per patch
  vector[N_patches] recovery_period;              // Posterior recovery period per patch

  // Compute the NGM for each patch pair
  for (p in 1:N_patches)
    for (j in 1:N_patches)
      NGM[p, j] = beta_mat[p, j] * N[p] / N[j] / gamma[j];

  // Compute the incubation and recovery periods for each patch
  for (p in 1:N_patches) {
    incubation_period[p] = 1.0 / sigma[p];
    recovery_period[p] = 1.0 / gamma[p];
    pred_incidence[p] = neg_binomial_2_rng(adjusted_incidence[p], phi);
  }
}