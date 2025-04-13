// Create the function to compute the SEIR dynamics

functions {
  vector seir_dynamics(vector state, int n_patches, vector beta, real sigma, real gamma, vector N) {
    vector[4 * n_patches] dstate;

    // Extract the compartments
    for (p in 1:n_patches) {
      real S = state[(p-1)*4 + 1];
      real E = state[(p-1)*4 + 2];
      real I = state[(p-1)*4 + 3];
      real R = state[(p-1)*4 + 4];
      
      // Force of infection (no coupling between patches)
      real foi = beta[p] * S * I / N[p];
      
      // SEIR equations
      dstate[(p-1)*4 + 1] = -foi;                     // dS/dt
      dstate[(p-1)*4 + 2] = foi - sigma * E;          // dE/dt
      dstate[(p-1)*4 + 3] = sigma * E - gamma * I;    // dI/dt
      dstate[(p-1)*4 + 4] = gamma * I;                // dR/dt
    }
    return dstate;
  }
}

data {
  int<lower=1> T;                 // Number of weeks
  int<lower=1> n_patches;         // Number of patches (countries)
  int<lower=0> I_obs[T, n_patches];        // Observed weekly incidence
  vector[n_patches] N;            // Population sizes
  real<lower=0> sigma;                     // Fixed rate: Exposed to Infectious
  real<lower=0> gamma;                     // Fixed rate: Infectious to Recovered
}

transformed data {
  int n_days = T * 7;             // Number of days (weekly data, 7 days per week)
  real dt = 1.0;                  // Time step (1 day)
  int steps_per_week = 7;         // Number of daily steps per week
}

parameters {
  vector<lower=0>[n_patches] beta;              // Transmission rates
  real<lower=0> phi;                            // Dispersion parameter for Negative Binomial
  vector<lower=0>[n_patches] E0;               // Initial Exposed
  vector<lower=0>[n_patches] I0;               // Initial Infectious
}

transformed parameters {
  vector[4 * n_patches] state[n_days + 1];     // State: S, E, I, R for each patch
  matrix[n_days, n_patches] new_infectious;     // Daily new infectious (sigma * E)
  
  // Initial conditions
  for (p in 1:n_patches) {
    state[1][(p-1)*4 + 1] = N[p] - E0[p] - I0[p];  // S0
    state[1][(p-1)*4 + 2] = E0[p];                 // E0
    state[1][(p-1)*4 + 3] = I0[p];                 // I0
    state[1][(p-1)*4 + 4] = 0;                     // R0
  }
  
  // Simulate the SEIR model using Euler integration
  for (t in 1:n_days) {
    vector[4 * n_patches] dstate = seir_dynamics(state[t], n_patches, beta, sigma, gamma, N);
    state[t + 1] = state[t] + dt * dstate;
    
    // Compute new infectious individuals (sigma * E)
    for (p in 1:n_patches) {
      real E = state[t][(p-1)*4 + 2];
      new_infectious[t, p] = sigma * E;
    }
  }
}

model {
  // Priors
  beta ~ lognormal(log(0.3), 0.5);              // Prior on transmission rates
  phi ~ exponential(0.1);                       // Prior on dispersion parameter
  E0 ~ lognormal(log(1), 1);                    // Prior on initial Exposed
  I0 ~ lognormal(log(1), 1);                    // Prior on initial Infectious
  
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
  matrix[T, n_patches] I_pred;                  // Predicted weekly incidence
  vector[n_patches] R0;                         // Basic reproduction number
  
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