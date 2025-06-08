functions {
  vector seir_metapop(
    real t,
    vector y,                      // state: [S1,E1,I1,R1, S2,E2,I2,R2, S3,E3,I3,R3]
    vector beta_mat,               // coupling matrix B_pj [3,3]
    real sigma,                  // sigma for each patch
    real gamma,                  // gamma for each patch
    vector N                       // population for each patch
  ) {
    int n_patches = 3;
    vector[12] dydt;
    for (p in 1:n_patches) {
      real S_p = y[4*(p-1)+1];
      real E_p = y[4*(p-1)+2];
      real I_p = y[4*(p-1)+3];
      real R_p = y[4*(p-1)+4];

      // Force of infection from all patches
      real inf_sum = 0;
      for (j in 1:n_patches) {
        real I_j = y[4*(j-1)+3];
        inf_sum += beta_mat[j] * S_p * I_j / N[j];
      }

      dydt[4*(p-1)+1] = -inf_sum;                        // dS_p/dt
      dydt[4*(p-1)+2] = inf_sum - sigma * E_p;        // dE_p/dt
      dydt[4*(p-1)+3] = sigma * E_p - gamma * I_p; // dI_p/dt
      dydt[4*(p-1)+4] = gamma * I_p;                  // dR_p/dt
    }
    return dydt;
  }
}

data {
  int<lower=1> N_patches;           // 3 (Guinea, Liberia, Sierra Leone)
  int<lower=1> n_weeks;             
  vector[4*N_patches] y0;           // Initial states for all patches [S0,E0,I0,R0,D0,...]
  real t0;                         
  array[n_weeks] real t;            
  vector[N_patches] N;              // Population for each patch
  array[N_patches, n_weeks] int<lower=0> cases; // Observed weekly incidence
  vector[N_patches] beta_data;   // Prior mean values for B_pj
  real sigma_data;             // Prior mean values for sigma
  real gamma_data;             // Prior mean values for gamma
  real<lower=0, upper=1> reporting_rate;    
}

parameters {
  vector<lower=0>[N_patches] beta_mat; // Transmission rates (to be estimated)
  real<lower=0> sigma;              // Progression rates (to be estimated)
  real<lower=0> gamma;              // Recovery rates (to be estimated)
  real<lower=0> phi_inv;                         // Overdispersion
}

transformed parameters {
  array[n_weeks] vector[5*N_patches] y;     // States over time
  array[N_patches, n_weeks] real weekly_incidence; // Incidence for each patch
  real<lower=0> phi = 1.0 / phi_inv;
  array[N_patches, n_weeks] real adjusted_incidence;

  y = ode_bdf(seir_metapop, y0, t0, t, beta_mat, sigma, gamma, N);

  for (p in 1:N_patches) {
    // Weekly incidence: number progressing E->I (new infections)
    weekly_incidence[p, 1] = sigma * y[1, 4*(p-1)+2];
    adjusted_incidence[p, 1] = fmax((reporting_rate * weekly_incidence[p, 1]), 0.000001);
    for (w in 2:n_weeks) {
      weekly_incidence[p, w] = sigma * y[w-1, 4*(p-1)+2];
      adjusted_incidence[p, w] = fmax((reporting_rate * weekly_incidence[p, w]), 0.000001);
    }
  }
}

model {
  // Priors centered on data
  for (p in 1:N_patches)
    beta_mat[p] ~ lognormal(log(beta_data[p]), 0.5);

  sigma ~ lognormal(log(sigma_data), 0.5);
  gamma ~ lognormal(log(gamma_data), 0.5);
  phi_inv ~ exponential(5);

  // Likelihood
  for (p in 1:N_patches)
    cases[p] ~ neg_binomial_2(adjusted_incidence[p], phi);
}

generated quantities {
  array[N_patches] real R0;
  real recovery_time;
  real incubation_period;
  array[N_patches, n_weeks] real pred_incidence;

  for (p in 1:N_patches) {
    // R0: sum all transmission rates from other patches to p, divided by gamma
    real beta_sum = 0;
    for (j in 1:N_patches) beta_sum += beta_mat[j];
    R0[p] = beta_sum / gamma;
    recovery_time = 1.0 / gamma;
    incubation_period = 1.0 / sigma;
    pred_incidence[p] = neg_binomial_2_rng(adjusted_incidence[p], phi);
  }
}