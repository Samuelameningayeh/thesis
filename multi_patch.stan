functions {
  vector seir(real t, vector y, array[] real theta, array[] real x_r, array[] int x_i) {
    int p = x_i[1]; // Patch index
    int N = x_i[2]; // Population size for patch p
    
    real sigma = theta[1]; // Estimated
    real gamma = theta[2]; // Estimated
    real alpha = theta[3];
    real beta = theta[3 + p]; // Patch-specific beta
    
    real S = y[1];
    real E = y[2];
    real I = y[3];
    real R = y[4];
    real D = y[5];
    
    vector[6] dydt;
    dydt[1] = -beta * I * S / N;              // dS/dt
    dydt[2] = beta * I * S / N - sigma * E;   // dE/dt
    dydt[3] = sigma * E - (gamma + alpha) * I;          // dI/dt
    dydt[4] = gamma * I;                      // dR/dt
    dydt[5] = alpha * I;                      // dD/dt
    dydt[6] = sigma * E;                      // Cumulative incidence
    
    return dydt;
  }
}

data {
  int<lower=1> P;               // Number of patches (3: Guinea, Liberia, Sierra Leone)
  int<lower=1> T;               // Number of time points
  real t0;                      // Initial time
  array[T] real ts;             // Observation times
  array[T, P] int<lower=0> cases; // Weekly incidence for each patch
  array[T, P] int<lower=0> deaths; // Weekly deaths for each patch
  array[P] int N;               // District population sizes
  array[P] real<lower=0> E0;    // Initial exposed
  array[P] real<lower=0> I0;    // Initial infected
  real<lower=0> gamma;          // Fixed recovery rate
  real<lower=0> sigma;          // Fixed incubation rate
  real<lower=0> alpha;          // Fixed mortality rate
}

transformed data {
  array[P] vector[6] y0;              // Initial state: [S, E, I, R, D, C]
  array[0] real x_r;
  // Set initial conditions
  for (p in 1:P) {
    y0[p, 1] = N[p] - E0[p] - I0[p]; // S(0)
    y0[p, 2] = E0[p];                // E(0)
    y0[p, 3] = I0[p];                // I(0)
    y0[p, 4] = 0;                    // R(0)
    y0[p, 5] = 0;                    // D(0)
    y0[p, 6] = 0;                    // Cumulative incidence
  }
}

parameters {
  array[P] real<lower=0> beta;  // Patch-specific transmission rates
  real<lower=0> phi_inv;        // Inverse dispersion parameter for cases
  real<lower=0> phi_d_inv;      // Inverse dispersion parameter for deaths
  array[P] real<lower=0, upper=1> rho;     // Patch-specific reporting rates for cases
  array[P] real<lower=0, upper=1> rho_d;   // Patch-specific reporting rates for deaths
}

transformed parameters {
  array[T, P] real incidence;
  array[T, P] real predicted_deaths;
  real<lower=0> phi = 1.0 / phi_inv;
  real<lower=0> phi_d = 1.0 / phi_d_inv;
  array[3 + P] real theta;
  
  // Set theta for ODE
  theta[1] = sigma;
  theta[2] = gamma;
  theta[3] = alpha;
  for (p in 1:P) {
    theta[3 + p] = beta[p];     // Patch-specific beta
  }
  
  // Solve ODE for each patch
  for (p in 1:P) {
    array[2] int x_i = {p, N[p]};            // Pass patch index and population
    array[T] vector[6] y_pred_p = ode_rk45(seir, y0[p], t0, ts, theta, x_r, x_i);
    
    // Compute incidence and predicted deaths for patch p
    incidence[1, p] = y_pred_p[1, 6];  // Initial cumulative incidence
    predicted_deaths[1, p] = y_pred_p[1, 5];  // Initial cumulative deaths
    for (t in 2:T) {
      incidence[t, p] = y_pred_p[t, 6] - y_pred_p[t-1, 6];  // New infections
      predicted_deaths[t, p] = y_pred_p[t, 5] - y_pred_p[t-1, 5];  // New deaths
    }
    for (t in 1:T) {
      incidence[t, p] = rho[p] * incidence[t, p];  // Adjust for case reporting
      predicted_deaths[t, p] = rho_d[p] * predicted_deaths[t, p];  // Adjust for death reporting
    }
  }
}

model {
  // Priors
  for (p in 1:P) {
    beta[p] ~ lognormal(log(0.3), 0.2);
    rho[p] ~ beta(2, 2);
    rho_d[p] ~ beta(2, 2);  // Prior for death reporting rate
  }
  phi_inv ~ exponential(5);
  phi_d_inv ~ exponential(5);
  
  // Likelihood for cases
  for (p in 1:P) {
    cases[:, p] ~ neg_binomial_2(incidence[:, p], phi);
  }
  
  // Likelihood for deaths
  for (p in 1:P) {
    deaths[:, p] ~ neg_binomial_2(predicted_deaths[:, p], phi_d);
  }
}

generated quantities {
  array[P] real R0;                     // Basic reproduction number
  array[T, P] real predicted_cases;     // Predicted cases
  
  for (p in 1:P) {
    R0[p] = beta[p] / (gamma + alpha);  // R0 = beta / (gamma + alpha)
  }
  
  for (p in 1:P) {
    for (t in 1:T) {
      predicted_cases[t, p] = neg_binomial_2_rng(incidence[t, p], phi);
    }
  }
}