functions {
  vector seir(real t,
             vector y, 
             real beta, 
             real sigma,
             real gamma,
             real N,
             vector r_ij,
             vector g_i) {

      vector[5] dydt;
      real S = y[1];
      real E = y[2];
      real I = y[3];
      real R = y[4];
      real incidence = 0.0;

      // Patch-specific dynamics
      dydt[1] = -beta * I * S / N + sum(r_ij[1:(num_elements(r_ij)-1)] * S) - g_i[1] * S;
      dydt[2] = beta * I * S / N - gamma * E + sum(r_ij[1:(num_elements(r_ij)-1)] * E) - g_i[1] * E;
      dydt[3] = gamma * E - sigma * I + sum(r_ij[1:(num_elements(r_ij)-1)] * I) - g_i[1] * I;
      dydt[4] = sigma * I + sum(r_ij[1:(num_elements(r_ij)-1)] * R) - g_i[1] * R;
      dydt[5] = gamma * E; // Track new infections for incidence

      return dydt;
  }
}
data {
  int<lower=1> n_patches;           // Number of patches
  int<lower=1> n_days;              // Number of days
  array[n_patches] vector[5] y0;    // Initial conditions for each patch
  real t0;                          // Initial time
  array[n_days] real t;             // Time points
  array[n_patches] int N;           // Population sizes for each patch
  array[n_patches, n_days] int<lower=0> cases; // Observed cases per patch
  array[n_patches, n_patches] real<lower=0> r_ij; // Movement rates between patches
  array[n_patches] real<lower=0> g_i; // Exit rates from patch i (same for S, E, I, R)
  array[N_countries] real beta_values; 
  array[N_countries] real sigma_values;
  array[N_countries] real gamma_values; 
  real<lower=0, upper=1> reporting_rate;
}
parameters {
  array[N_countries] real<lower=0> beta;       // Country-specific transmission rates
  array[N_countries] real<lower=0> sigma;                     // Shared recovery rate
  //real<lower=0> alpha;
  array[N_countries] real<lower=0> gamma;  
  real<lower=0> phi_inv;
}
transformed parameters{
  array[N_countries, n_weeks] real adjusted_incidence;
  array[n_patches, n_days] vector[5] y;
  array[n_patches, n_days] incidence;
  real<lower=0> phi = 1. / phi_inv;
  //real<lower=0> reporting_rate = 1;

  for (i in 1:n_patches) {
    y[i] = ode_bdf(seir, y0[i], t0, t, beta[i], sigma[i], gamma[i], N[i], to_vector(r_ij[i]), to_vector({g_i[i], g_i[i], g_i[i], g_i[i]}));
    incidence[i, 1] = y[i, 1, 5];
    for (j in 2:n_days)
      incidence[i, j] = y[i, j, 5] - y[i, j-1, 5];
      adjusted_incidence[i, j] = reporting_rate * incidence[i, j]+0.000005;
  }
}
model {
  // Priors
 for (p in 1:N_countries) {
    beta[p] ~ lognormal(log(beta_values[p]), 0.5);
    //alpha ~ lognormal(log(0.3), 0.2);
    sigma[p] ~ lognormal(log(sigma_values[p]), 0.5);      // Prior mean: 10-day recovery period
    gamma[p] ~ lognormal(log(gamma_values[p]), 0.5);      // Prior mean: 7-day incubation period
  }    // Prior mean: 7-day infectious period
  phi_inv ~ exponential(2);

  // Sampling distribution
  for (i in 1:n_patches)
    cases[i] ~ neg_binomial_2(adjusted_incidence[i], phi);
}
generated quantities {
  array[N_countries] real R0;   // Country-specific R0
  array[N_countries] real recovery_time;        // Shared recovery time
  array[N_countries] real incubation_period;    // Shared incubation period
  array[N_countries, n_days] real pred_incidence; // Predicted cases

  for (c in 1:N_countries) {
    R0[c] = (beta[c] / gamma[c]);  // R0 including mortality
    recovery_time[c] = (1.0 / sigma[c]);        // Shared recovery time
    incubation_period[c] = (1.0 / gamma[c]); 
  }
    for (i in 1:n_patches){
      pred_incidence[i] = neg_binomial_2_rng(adjusted_incidence[i], phi);
    }
}