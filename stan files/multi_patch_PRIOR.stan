functions {
  vector seir(real t,
              vector y,
              real beta,
              real sigma,
              real gamma,
              real alpha,
              real N) {
    //vector[5] dydt;
    vector[6] dydt;
    real S = y[1];
    real E = y[2];
    real I = y[3];
    real R = y[4];
    real D = y[5];

    dydt[1] = -beta * I * S / N;              // dS/dt (weekly)
    dydt[2] = beta * I * S / N - sigma * E;   // dE/dt (weekly)
    dydt[3] = sigma * E - (gamma+alpha) * I;          // dI/dt (weekly)
    dydt[4] = gamma * I;                      // dR/dt (weekly)
    dydt[5] = alpha * I;
    dydt[6] = sigma * E;                      // Cumulative incidence (weekly)

    return dydt;
  }
}

data {
  int<lower=1> N_countries;                // Number of countries (3)
  int<lower=1> n_weeks;                    // Number of weeks
  array[N_countries] vector[6] y0;         // Initial conditions for each country
  real t0;                                 // Initial time
  array[n_weeks] real t;                   // Time points (weekly: 0, 1, 2, ...)
  array[N_countries] int N;                // Population sizes
  array[N_countries, n_weeks] int<lower=0> cases; // Observed weekly cases for each country
  array[N_countries] real beta_values;     // Daily beta values
  array[N_countries] real sigma_values;    // Daily sigma values
  array[N_countries] real gamma_values;    // Daily gamma values
  array[N_countries] real alpha_values;    // Daily alpha values
  real<lower=0, upper=1> reporting_rate;
}

parameters {
  array[N_countries] real<lower=0> beta;       // Country-specific transmission rates (weekly)
  array[N_countries] real<lower=0> sigma;      // Country-specific recovery rate (weekly)
  array[N_countries] real<lower=0> gamma;      // Country-specific progression rate (E to I, weekly)
  array[N_countries] real<lower=0> alpha;      // Country-specific death rate (weekly)
  array[N_countries] real<lower=0> phi_inv;                       // Negative binomial overdispersion
}

transformed parameters {
  array[N_countries, n_weeks] vector[6] y;   // Weekly SEIR states for each country
  array[N_countries, n_weeks] real weekly_incidence; // Weekly incidence for each country
  array[N_countries, n_weeks] real adjusted_incidence;
  array[N_countries] real<lower=0> phi;

  for (c in 1:N_countries) {
    phi[c] = 1.0 / phi_inv[c];
  }

  // Solve ODE at weekly intervals with rescaled rates
  for (c in 1:N_countries) {
    y[c] = ode_rk45(seir, y0[c], t0, t, beta[c], sigma[c], gamma[c], alpha[c], N[c]);
    
    // Compute weekly incidence
    weekly_incidence[c, 1] = y[c, 1, 6];         // Initial incidence
    adjusted_incidence[c,1] = (reporting_rate * weekly_incidence[c,1])+0.00005;

    for (w in 2:n_weeks) {
      weekly_incidence[c, w] = y[c, w, 6] - y[c, w-1, 6];
      adjusted_incidence[c, w] = (reporting_rate * weekly_incidence[c, w])+0.00005;
    }
  }
}

model {
  // PRIORS (rescaled to weekly rates)
  for (p in 1:N_countries) {
    beta[p] ~ lognormal(log(beta_values[p]), 0.3);    // Weekly beta = daily beta / 7
    sigma[p] ~ lognormal(log(sigma_values[p]), 0.3);  // Weekly sigma = daily sigma / 7
    gamma[p] ~ lognormal(log(gamma_values[p]), 0.3);  // Weekly gamma = daily gamma / 7
    alpha[p] ~ lognormal(log(alpha_values[p]), 0.3);  // Weekly alpha = daily alpha / 7
    phi_inv[p] ~ exponential(5);
  }
}

generated quantities {
  array[N_countries] real R0;   // Country-specific R0 (weekly)
  array[N_countries] real recovery_time;        // Recovery time (weeks)
  array[N_countries] real incubation_period;    // Incubation period (weeks)
  array[N_countries, n_weeks] real pred_incidence; // Predicted weekly cases

  for (c in 1:N_countries) {
    R0[c] = (beta[c] / (gamma[c]+alpha[c]));  // R0 based on weekly rates
    recovery_time[c] = (1.0 / gamma[c]);      // Weeks
    incubation_period[c] = (1.0 / sigma[c]);    // Weeks
  }

  for (c in 1:N_countries) {
    pred_incidence[c] = neg_binomial_2_rng(adjusted_incidence[c], phi[c]);
  }
}