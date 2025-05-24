functions {
  vector seir(real t, vector y, array[] real theta, 
             real beta, 
             real sigma,
             real gamma,
             real N, 
             array[] int x_i) {

    int p = x_i[1]; // Patch index
    //int N = x_i[3]; // Population size for patch p
    
    real S = y[1];
    real E = y[2];
    real I = y[3];
    real R = y[4];
    //real D = y[5];
    
    vector[5] dydt;
    dydt[1] = -beta * I * S / N;              // dS/dt
    dydt[2] = beta * I * S / N - sigma * E;   // dE/dt
    dydt[3] = sigma * E - (gamma) * I;          // dI/dt
    dydt[4] = gamma * I;                      // dR/dt
    //dydt[5] = alpha * I;                      // dD/dt
    dydt[5] = sigma * E;                  // Cumulative incidence
    
    return dydt;
  }
}

data {
  int<lower=1> P;               // Number of patches (3: Guinea, Liberia, Sierra Leone)
  int<lower=1> T;               // Number of time points
  real t0;                      // Initial time
  array[P] vector[5] y0; 
  array[T] real ts;             // Observation times
  array[T, P] int<lower=0> cases; // Weekly incidence for each patch
  array[P] int N;               // District population sizes
  real<lower=0> gamma;          // Fixed recovery rate
  real<lower=0> sigma;          // Fixed incubation rate
  //real<lower=0> alpha;          // Fixed mortality rate
}

parameters {
  array[P] real<lower=0> beta;  // Patch-specific transmission rates
  real<lower=0> phi_inv;        // Inverse dispersion parameter for cases
  real<lower=0> sigma;
  real<lower=0> gamma;

  //real<lower=0> phi_d_inv;      // Inverse dispersion parameter for deaths
  //array[P] real<lower=0, upper=1> rho;     // Patch-specific reporting rates for cases
  //array[P] real<lower=0, upper=1> rho_d;   // Patch-specific reporting rates for deaths
}

transformed parameters {
  array[T, P] real incidence;
  real<lower=0> phi = 1.0 / phi_inv;
  
  for (p in 1:P) {
    array[2] int x_i = {p, N[p]};            // Pass patch index and population
    array[T] vector[5] y_pred_p = ode_rk45(seir, y0[p], t0, ts, theta, x_r, x_i);
    
    // Compute incidence, predicted deaths, and store S(t)
    incidence[1, p] = y_pred_p[1, 5];  // Initial cumulative incidence

    for (t in 2:T) {
      incidence[t, p] = y_pred_p[t, 5] - y_pred_p[t-1, 5];         // New infections
    }

    for (t in 1:T) {
      incidence[t, p] = incidence[t, p];   // Adjust for case reporting
    }
  }
}

model {
  // Priors
  beta ~ lognormal(log(0.3), 0.2);      // Infection rate prior
  sigma ~ lognormal(log(1.0 / 7), 0.2);   // Prior mean: 5-day incubation period
  gamma ~ lognormal(log(1.0/10), 0.3);   // Prior mean: 7-day infectious period 
  phi_inv ~ exponential(5);
  
  // Likelihood for cases
  for (p in 1:P) {
    cases[:, p] ~ neg_binomial_2(0.00001+incidence[:, p], phi);
  }
}

generated quantities {
  array[P] real R0;                     // Basic reproduction number
  //array[T, P] real Rt;                  // Effective reproduction number
  array[T, P] real predicted_cases;     // Predicted cases
  
  for (p in 1:P) {
    R0[p] = beta[p] / (gamma);  // R0 = beta / (gamma + alpha)
    
    for (t in 1:T){
      //Rt[t, p] = R0[p] * S[t, p] / N[p];  // Rt = R0 * S(t) / N
      predicted_cases[t, p] = neg_binomial_2_rng(0.00001 + incidence[t, p], phi);
    }
  }
}