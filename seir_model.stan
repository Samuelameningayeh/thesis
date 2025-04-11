functions {
  real[] seir(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
    
    real S = y[1];
    real E = y[2];
    real I = y[3];
    real R = y[4];
    
    real N = x_i[1];

    real beta = theta[1];
    real sigma = theta[2];
    real gamma = theta[3];
    
    real dS_dt = -beta * S * I / N;
    real dE_dt = beta * S * I / N - sigma * E;
    real dI_dt = sigma * E - gamma * I;
    real dR_dt = gamma * I;
    
    return {dS_dt, dE_dt, dI_dt, dR_dt};
  
  }

}

data {
  int<lower=1> T;           // Number of time points
  real t0;                  // Initial time
  real ts[T];               // Observation times
  int y[T];                 // Observed cases
  int N;                   // Population size
  real I0;                  // Initial infected

}

transformed data {
  real y0[4];               // Initial state: [S, E, I, R]
  real x_r[1] = {0};        // Real data for ODE (population size)
  int x_i[1] = {N};         // Integer data for ODE (unused here)
  y0[1] = N - I0;           // S(0)
  y0[2] = 0;                // E(0)
  y0[3] = I0;               // I(0)
  y0[4] = 0;                // R(0)

}

parameters {
  real<lower=0> beta;       // Infection rate
  real<lower=0> sigma;      // Rate of becoming infectious
  real<lower=0> gamma;      // Recovery rate
  real<lower=0, upper=1> rho; // Reporting rate

}

model {
  // Priors
  beta ~ normal(0.3, 0.1);  // Weak prior around 0.5
  sigma ~ normal(0.2, 0.05); // Around 5-day latent period
  gamma ~ normal(0.1, 0.05); // Around 10-day infectious period
  rho ~ beta(1, 1);         // Uniform prior

  // ODE solution
  real theta[3] = {beta, sigma, gamma};
  real y_hat[T, 4] = integrate_ode_rk45(seir, y0, t0, ts, theta, x_r, x_i);

  // Likelihood: observed cases ~ Poisson(rho * sigma * E)
  for (t in 1:T) {
    y[t] ~ poisson(rho * sigma * y_hat[t, 2]); // y_hat[t, 2] is E(t)
    //y[t] ~ neg_binomial_2(col(to_matrix(y), 2), phi);
  
  }

}

generated quantities {
  real y_pred[T, 4];        // Predicted states
  real theta[3] = {beta, sigma, gamma};
  real R0 = beta / gamma;
  real recovery_time = 1 / gamma;
  y_pred = integrate_ode_rk45(seir, y0, t0, ts, theta, x_r, x_i);

}