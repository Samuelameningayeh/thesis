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

    dydt[1] = -beta * I * S / N;              // dS/dt
    dydt[2] = beta * I * S / N - gamma * E;   // dE/dt
    dydt[3] = gamma * E - sigma * I;          // dI/dt
    dydt[4] = sigma * I;                      // dR/dt
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
}
parameters {
  vector<lower=0>[N_countries] beta;       // Country-specific transmission rates
  real<lower=0> sigma;                     // Shared recovery rate
  real<lower=0> gamma;                     // Shared progression rate (E to I)
  real<lower=0> phi_inv;                   // Negative binomial overdispersion
}
transformed parameters {
    real<lower=0,upper=1> reporting_rate;    
  array[N_countries, n_days] vector[5] y;   // SEIR states for each country
  array[N_countries, n_days] real incidence; // Incidence for each country
  real<lower=0> phi = 1.0 / phi_inv;       // Negative binomial dispersion
  
  reporting_rate = 0.8;

  for (c in 1:N_countries) {
    // Solve ODE for each country
    y[c] = ode_rk45(seir, y0[c], t0, t, beta[c], sigma, gamma, N[c] + 0.0);
    // Compute incidence
    incidence[c, 1] = y[c, 1, 5];         // Initial incidence
    for (i in 2:n_days) {
      incidence[c, i] = y[c, i, 5] - y[c, i-1, 5];
    }
  }
}
model {
  // Priors
  beta ~ lognormal(log(0.35), 0.5);        // Infection rate prior
  sigma ~ lognormal(log(1.0/7), 0.3);      // Prior mean: 4-day recovery period
  gamma ~ lognormal(log(1.0/10), 0.3);      // Prior mean: 6-day incubation period
  phi_inv ~ exponential(5);
  //reporting_rate ~ beta(8, 2);             // Centered around 0.8

  // Likelihood
  for (c in 1:N_countries) {
    for (i in 1:n_days) {
      cases[c, i] ~ neg_binomial_2((reporting_rate * incidence[c, i]+0.0001), phi);
    }
  }
}
generated quantities {
  vector[N_countries] R0 = beta / sigma;   // Country-specific R0
  real recovery_time = 1.0 / sigma;        // Shared recovery time
  real incubation_period = 1.0 / gamma;    // Shared incubation period
  array[N_countries, n_days] real pred_incidence; // Predicted cases

  for (c in 1:N_countries) {
    for (i in 1:n_days) {
      pred_incidence[c, i] = neg_binomial_2_rng((reporting_rate * incidence[c, i]+0.0001), phi);
    }
  }
}