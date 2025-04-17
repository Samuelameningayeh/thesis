// Create the function for the SEIR ODEs
functions {
  vector seir_ode(real time, vector state, real beta, real sigma, real gamma, real N) {
    vector[4] dstate;
    real S = state[1];
    real E = state[2];
    real I = state[3];
    real R = state[4];

    real foi = beta * S * I / N;

    dstate[1] = -foi;       // dS/dt
    dstate[2] = foi - sigma * E;   // dE/dt
    dstate[3] = sigma * E - gamma * I; // dI/dt
    dstate[4] = gamma * I;     // dR/dt
    return dstate;
  }
}

data {
  int<lower=1> T;                     // Number of weeks
  int<lower=1> n_patches;             // Number of patches (countries)
  array[T, n_patches] int<lower=0> I_obs; // Observed weekly incidence
  vector[n_patches] N;                 // Population sizes
  vector<lower=0>[n_patches] E0;        // Initial Exposed
  vector<lower=0>[n_patches] I0;        // Initial Infectious
}

transformed data {
  int n_days = T * 7;               // Number of days (weekly data, 7 days per week)
  real dt = 1.0;                    // Time step (1 day)
  int steps_per_week = 7;            // Number of daily steps per week
}

parameters {
  vector<lower=0>[n_patches] beta;    // Transmission rates
  real<lower=0> phi;                // Dispersion parameter for Negative Binomial
  real<lower=0> sigma;              // Fixed rate: Exposed to Infectious
  real<lower=0> gamma;              // Fixed rate: Infectious to Recovered
}

transformed parameters {
  array[n_days + 1] vector[4 * n_patches] state; // State: S, E, I, R for each patch
  matrix[n_days, n_patches] new_infectious;     // Daily new infectious (sigma * E)

  // Initial conditions
  for (p in 1:n_patches) {
    state[1][(p-1)*4 + 1] = N[p] - E0[p] - I0[p]; // S0
    state[1][(p-1)*4 + 2] = E0[p];             // E0
    state[1][(p-1)*4 + 3] = I0[p];             // I0
    state[1][(p-1)*4 + 4] = 0;                 // R0
  }

  real t0 = 0;
  array[n_days] real ts;
  for (i in 1:n_days) {
    ts[i] = i;
  }

  for (p in 1:n_patches) {
    vector[4] initial_state = state[1][((p-1)*4 + 1):(p*4)];
    array[n_days, 4] real sol = integrate_ode_rk45(seir_ode, initial_state, t0, ts, beta[p], sigma, gamma, N[p]);
    for (t in 1:n_days) {
      state[t + 1][(p-1)*4 + 1] = sol[t][1]; // S
      state[t + 1][(p-1)*4 + 2] = sol[t][2]; // E
      state[t + 1][(p-1)*4 + 3] = sol[t][3]; // I
      state[t + 1][(p-1)*4 + 4] = sol[t][4]; // R
      new_infectious[t, p] = sigma * sol[t][2]; // Daily new infectious
    }
  }
}

model {
  // Priors
  beta ~ lognormal(log(0.3), 0.5);         // Prior on transmission rates
  phi ~ exponential(0.1);                   // Prior on dispersion parameter
  sigma ~ normal(0.2, 0.05);                // Around 5-day latent period
  gamma ~ normal(0.1, 0.05);                // Around 10-day infectious period
  E0 ~ lognormal(log(1), 1);               // Prior on initial Exposed
  I0 ~ lognormal(log(1), 1);               // Prior on initial Infectious

  // Likelihood
  for (t in 1:T) {
    for (p in 1:n_patches) {
      real weekly_incidence = 0;
      // Sum daily new infectious over the week
      for (d in ((t-1)*steps_per_week + 1):(t*steps_per_week)) {
        weekly_incidence += new_infectious[d, p];
      }
      I_obs[t, p] ~ neg_binomial_2(weekly_incidence, phi);
    }
  }
}

generated quantities {
  matrix[T, n_patches] I_pred;       // Predicted weekly incidence
  vector[n_patches] R0;             // Basic reproduction number

  // Compute predicted incidence
  for (t in 1:T) {
    for (p in 1:n_patches) {
      real weekly_incidence = 0;
      for (d in ((t-1)*steps_per_week + 1):(t*steps_per_week)) {
        weekly_incidence += new_infectious[d, p];
      }
      I_pred[t, p] = weekly_incidence;
    }
  }

  // Compute R0 for each patch
  for (p in 1:n_patches) {
    R0[p] = beta[p] / gamma;
  }
}