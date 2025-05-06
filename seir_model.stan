functions {
  vector seir(real t, vector y, array[] real theta, array[] real x_r, array[] int x_i) {
    
    vector[5] dydt;

    real S = y[1];
    real E = y[2];
    real I = y[3];
    real R = y[4];

    real N = x_i[1];

    real beta = theta[1];
    real sigma = theta[2];
    real gamma = theta[3];

    dydt[1] = -beta * I * S / N;                 // Change in susceptible
    dydt[2] = beta * I * S / N - sigma * E;      // Change in exposed
    dydt[3] = sigma * E - gamma * I;             // Change in infectious
    dydt[4] = gamma * I;                         // Change in recovered
    dydt[5] = sigma * E;   
      
    return dydt;
  }
}

data {
  int<lower=1> T;         // Number of time points
  real t0;               // Initial time
  array[T] real ts;       // Observation times
  array[T] int<lower=0> cases; // Observed cases 
  int N;                 // Population size
  real I0;               // Initial infected
}

transformed data {
  vector[5] y0;           // Initial state: [S, E, I, R]
  array[0] real x_r; 
  array[1] int x_i = {N};  
  y0[1] = N - I0;       // S(0)
  y0[2] = 0;           // E(0)
  y0[3] = I0;           // I(0)
  y0[4] = 0;           // R(0)
  y0[5] = 0;
  }

parameters {
  real<lower=0> beta;      // Infection rate
  real<lower=0> sigma;     // Rate of becoming infectious
  real<lower=0> gamma;     // Recovery rate
  real<lower=0> phi_inv;
  real<lower=0, upper=1> rho; // Reporting rate
}

transformed parameters{
  // ODE solution
  vector[T] incidence;
  real<lower=0> phi = 1. / phi_inv;
  array[3] real theta = {beta, sigma, gamma};
  array[T] vector[5] y_pred = ode_rk45(seir, y0, t0, ts, theta, x_r, x_i);

  // number of incidence for a given time
  incidence[1] = y_pred[1, 5] - 0;
  for (i in 2:T)
    incidence[i] = y_pred[i, 5] - y_pred[i-1, 5];

  // Apply reporting rate
  incidence = rho * incidence;

}

model {
  // Priors
  beta ~ lognormal(log(0.3), 0.2);      // Infection rate prior
  sigma ~ lognormal(log(1.0 / 5), 0.2);   // Prior mean: 5-day incubation period
  gamma ~ lognormal(log(1.0/10), 0.3);   // Prior mean: 7-day infectious period
  phi_inv ~ exponential(5);             // Dispersion parameter
  rho ~ beta(2, 2);

  // Likelihood: observed infected ~ Poisson(I)
  cases ~ neg_binomial_2(rho*incidence, phi);
}

generated quantities {
  real R0 = beta / gamma;
  real recovery_time = 1 / sigma;
  real incubation_period = 1 / gamma;
  
  array[T] real predicted_cases; // Predicted number of observed cases
  //predicted_cases = neg_binomial_2_rng(incidence, phi);
  for (i in 1:T)
    predicted_cases[i] = neg_binomial_2_rng(rho*incidence[i], phi);

}