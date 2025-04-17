functions {
  vector seir(real t, vector y, array[] real theta, array[] real x_r, array[] int x_i) {
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

    return to_vector({dS_dt, dE_dt, dI_dt, dR_dt});
  }
}

data {
  int<lower=1> T;         // Number of time points
  real t0;               // Initial time
  array[T] real ts;       // Observation times
  array[T] int y;         // Observed cases (Assuming these are observed infected individuals)
  int N;                 // Population size
  real I0;               // Initial infected
}

transformed data {
  vector[4] y0;           // Initial state: [S, E, I, R]
  array[0] real x_r; 
  array[1] int x_i = {N};  
  y0[1] = N - I0;       // S(0)
  y0[2] = 0;           // E(0)
  y0[3] = I0;           // I(0)
  y0[4] = 0;           // R(0)
}

parameters {
  real<lower=0> beta;      // Infection rate
  real<lower=0> sigma;     // Rate of becoming infectious
  real<lower=0> gamma;     // Recovery rate
  real<lower=0> phi;
  real<lower=0, upper=1> rho; // Reporting rate (Assuming observed cases are a fraction of actual infected)
}

model {
  // Priors
  beta ~ normal(0.5, 0.1);    // Weak prior around 0.5
  sigma ~ normal(0.3, 0.05);    // Around 5-day latent period
  gamma ~ lognormal(log(0.2), 0.5);     // Around 10-day infectious period
  rho ~ beta(1, 1);          // Uniform prior for reporting rate
  phi ~ exponential(0.5);

  // ODE solution
  array[3] real theta = {beta, sigma, gamma};
  array[T] vector[4] y_hat = ode_rk45(seir, y0, t0, ts, theta, x_r, x_i);

  // Likelihood: observed infected ~ Poisson(I)

  for (t in 1:T) {
    y[t] ~ poisson(rho*y_hat[t][3]); 
    //y[t] ~ neg_binomial_2(y_hat[t][3], phi);
  }
}

generated quantities {
  array[3] real theta = {beta, sigma, gamma};
  array[T] vector[4] y_pred = ode_rk45(seir, y0, t0, ts, theta, x_r, x_i);  // Predicted states
  
  real R0 = beta / gamma;
  real recovery_time = 1 / gamma;
  
  array[T] real predicted_cases; // Predicted number of observed cases
  for (t in 1:T) {
    predicted_cases[t] = poisson_rng(rho*y_pred[t][3]); //neg_binomial_2_rng(y_pred[t][3], phi); //
  }
}