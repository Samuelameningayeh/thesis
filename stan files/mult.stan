functions {
  vector seir(real t,
              vector y,
              real beta,
              real sigma,
              real gamma,
              real N) {
    vector[5] dydt;
    real S = y[1];
    real E = y[2];
    real I = y[3];
    real R = y[4];
    //real D = y[5]

    dydt[1] = -beta * I * S / N;              // dS/dt
    dydt[2] = beta * I * S / N - gamma * E;   // dE/dt
    dydt[3] = gamma * E - (sigma) * I;          // dI/dt
    dydt[4] = sigma * I;                      // dR/dt
    //dydt[5] = alpha * I;  
    dydt[5] = gamma * E;                      // Cumulative incidence

    return dydt;
  }
}
data {
  int<lower=1> N_countries;                // Number of countries (3)
  int<lower=1> n_days;                     // Number of time points
  array[N_countries] vector[5] y0;         // Initial conditions for each country
  real t0;                                 // Initial time
  array[n_days] real t;                    // Time points
  array[N_countries] int N;                // Population sizes
  array[N_countries, n_days] int<lower=0> cases; // Observed cases for each country
  array[N_countries] real beta_values; 
}
parameters {
  //real<lower=0,upper=1> reporting_rate; 

  array[N_countries] real beta;       // Country-specific transmission rates
  real<lower=0> sigma;                     // Shared recovery rate
  //real<lower=0> alpha;
  real<lower=0> gamma;                     // Shared progression rate (E to I)
  real<lower=0> phi_inv;                   // Negative binomial overdispersion
}
transformed parameters { 
  real<lower=0, upper=1> reporting_rate;
  array[N_countries, n_days] vector[5] y;   // SEIR states for each country
  array[N_countries, n_days] real incidence; // Incidence for each country
  real<lower=0> phi = 1.0 /phi_inv;       // Negative binomial dispersion
  
  reporting_rate = 0.8;

  for (c in 1:N_countries) {
    // Solve ODE for each country
    y[c] = ode_rk45(seir, y0[c],t0, t, beta[c], sigma, gamma, N[c]);
    
    // Compute incidence
    incidence[c, 1] = y[c, 1, 5];         // Initial incidence
    for (i in 2:n_days) {
      incidence[c, i] = y[c, i, 5] - y[c, i-1, 5];
    }
  }
}
model {
  // Priors
  for (p in 1:N_countries) {
    beta[p] ~ lognormal(log(beta_values[p]), 0.5);
    //alpha ~ lognormal(log(0.3), 0.2);
    sigma ~ lognormal(log(1.0/10), 0.5);      // Prior mean: 10-day recovery period
    gamma ~ lognormal(log(1.0/7), 0.5);      // Prior mean: 7-day incubation period
    phi_inv ~ exponential(5);
    reporting_rate ~ beta(2, 2);             // Centered around 0.8
  }
  //alpha ~ lognormal(log(0.3*7), 0.2);
  //sigma ~ lognormal(log(1.0/10*7), 0.5);      // Prior mean: 10-day recovery period
  //gamma ~ lognormal(log(1.0/7*7), 0.5);      // Prior mean: 7-day incubation period
  //phi_inv ~ exponential(5*7);
  //reporting_rate ~ beta(2, 2);             // Centered around 0.8

  // Likelihood
  for (c in 1:N_countries) {
    for (i in 1:n_days) {
      cases[c, i] ~ neg_binomial_2(reporting_rate*(incidence[c, i]+0.000001), phi);
    }
  }
}
generated quantities {
  array[N_countries] real R0;   // Country-specific R0
  real recovery_time = 1.0 / sigma*7;        // Shared recovery time
  real incubation_period = 1.0 / gamma*7;    // Shared incubation period
  array[N_countries, n_days] real pred_incidence; // Predicted cases

  for (c in 1:N_countries) {
    R0[c] = beta[c]*7 / gamma*7;  // R0 including mortality
    
    for (i in 1:n_days) {
      pred_incidence[c, i] = neg_binomial_2_rng(reporting_rate*(incidence[c, i]+0.000001), phi);
    }
  }
}