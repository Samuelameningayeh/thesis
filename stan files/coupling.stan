functions {
  vector seir_patch(real t, vector y, array[] real theta, array[] real x_r, array[] int x_i) {
    int P = x_i[1];               // Number of patches
    array[P] int N = x_i[2:P+1];  // Population sizes
    
    real sigma = theta[1];        // Incubation rate
    real gamma = theta[2];        // Recovery rate
    real mu = theta[3];           // Mortality rate
    array[P] real beta = theta[4:3+P];  // Transmission rates
    
    // Mobility matrix (passed as parameter)
    matrix[P, P] mobility = to_matrix(theta[4+P:], P, P);
    
    vector[6 * P] dydt;  // Derivatives for S, E, I, R, D, C
    
    for (p in 1:P) {
      real S_p = y[1 + 6*(p-1)];
      real E_p = y[2 + 6*(p-1)];
      real I_p = y[3 + 6*(p-1)];
      real R_p = y[4 + 6*(p-1)];
      real D_p = y[5 + 6*(p-1)];
      real C_p = y[6 + 6*(p-1)];  // Cumulative infections
      
      real new_infections = beta[p] * S_p * I_p / N[p];
      real new_infectious = sigma * E_p;
      real new_deaths = mu * I_p;
      
      real dS_in = 0.0, dE_in = 0.0, dI_in = 0.0, dR_in = 0.0, dD_in = 0.0, dC_in = 0.0;
      real dS_out = 0.0, dE_out = 0.0, dI_out = 0.0, dR_out = 0.0, dD_out = 0.0, dC_out = 0.0;
      
      for (k in 1:P) {
        if (k != p) {
          dS_out += mobility[p, k] * S_p;
          dE_out += mobility[p, k] * E_p;
          dI_out += mobility[p, k] * I_p;
          dR_out += mobility[p, k] * R_p;
          dD_out += mobility[p, k] * D_p;
          dC_out += mobility[p, k] * C_p;
          
          real S_k = y[1 + 6*(k-1)];
          real E_k = y[2 + 6*(k-1)];
          real I_k = y[3 + 6*(k-1)];
          real R_k = y[4 + 6*(k-1)];
          real D_k = y[5 + 6*(k-1)];
          real C_k = y[6 + 6*(k-1)];
          
          dS_in += mobility[k, p] * S_k;
          dE_in += mobility[k, p] * E_k;
          dI_in += mobility[k, p] * I_k;
          dR_in += mobility[k, p] * R_k;
          dD_in += mobility[k, p] * D_k;
          dC_in += mobility[k, p] * C_k;
        }
      }
      
      dydt[1 + 6*(p-1)] = -new_infections + dS_in - dS_out;  // dS/dt
      dydt[2 + 6*(p-1)] = new_infections - new_infectious + dE_in - dE_out;  // dE/dt
      dydt[3 + 6*(p-1)] = new_infectious - (gamma + mu) * I_p + dI_in - dI_out;  // dI/dt
      dydt[4 + 6*(p-1)] = gamma * I_p + dR_in - dR_out;  // dR/dt
      dydt[5 + 6*(p-1)] = new_deaths + dD_in - dD_out;  // dD/dt
      dydt[6 + 6*(p-1)] = new_infectious + dC_in - dC_out;  // dC/dt
    }
    
    return dydt;
  }
}

data {
  int<lower=1> P;               // Number of patches
  int<lower=1> T;               // Time points
  real t0;                      // Initial time
  array[T] real ts;             // Observation times
  array[T, P] int<lower=0> cases; // Observed incidence
  array[P] int<lower=1> N;      // Population sizes
  array[P] real<lower=0> E0;    // Initial exposed
  array[P] real<lower=0> I0;    // Initial infected
}

transformed data {
  array[6 * P] real y0;         // Initial state (S1, E1, I1, R1, D1, C1, ..., SP, EP, IP, RP, DP, CP)
  array[0] real x_r;            // No real-valued auxiliary data
  array[P + 1] int x_i;         // x_i[1] = P, x_i[2:P+1] = N[1:P]
  
  x_i[1] = P;
  for (p in 1:P) {
    x_i[p + 1] = N[p];
    y0[1 + 6*(p-1)] = N[p] - E0[p] - I0[p];  // S_p(0)
    y0[2 + 6*(p-1)] = E0[p];                 // E_p(0)
    y0[3 + 6*(p-1)] = I0[p];                 // I_p(0)
    y0[4 + 6*(p-1)] = 0;                     // R_p(0)
    y0[5 + 6*(p-1)] = 0;                     // D_p(0)
    y0[6 + 6*(p-1)] = 0;                     // C_p(0)
  }
}

parameters {
  real<lower=0> sigma;          // Incubation rate
  real<lower=0> gamma;          // Recovery rate
  real<lower=0> mu;             // Mortality rate
  array[P] real<lower=0> beta;  // Transmission rates
  matrix<lower=0>[P, P] mobility;  // Mobility matrix
  real<lower=0> phi_inv;        // Inverse dispersion
  array[P] real<lower=0, upper=1> rho;  // Reporting rates
}

transformed parameters {
  array[T, P] real incidence;
  real<lower=0> phi = 1.0 / phi_inv;
  array[3 + P + P*P] real theta;  // theta = [sigma, gamma, mu, beta[1:P], mobility[P, P]]
  
  theta[1] = sigma;
  theta[2] = gamma;
  theta[3] = mu;
  for (p in 1:P) {
    theta[3 + p] = beta[p];
  }
  for (i in 1:P) {
    for (j in 1:P) {
      theta[3 + P + (i-1)*P + j] = mobility[i, j];
    }
  }
  
  // Solve ODE for all patches
  array[T] vector[6 * P] y_pred = ode_rk45(seir_patch, to_vector(y0), t0, ts, theta, x_r, x_i);
  
  // Compute incidence as new infections scaled by reporting rate
  for (p in 1:P) {
    for (t in 1:T) {
      real E_p = y_pred[t, 2 + 6*(p-1)];  // Exposed at time t
      incidence[t, p] = sigma * E_p * rho[p];  // Incidence = new infectious cases * reporting rate
    }
  }
}

model {
  // Priors
  sigma ~ lognormal(log(1.0/7), 0.2);  // Latent period ~5 days
  gamma ~ lognormal(log(1.0/10), 0.3); // Infectious period ~10 days
  mu ~ lognormal(log(0.1), 0.2);       // Mortality rate ~50% over 10 days
  for (p in 1:P) {
    beta[p] ~ lognormal(log(0.3), 0.2);
    rho[p] ~ beta(1, 1);  // Uniform prior on reporting rate
  }
  for (i in 1:P) {
    for (j in 1:P) {
      mobility[i, j] ~ exponential(1);  // Sparse mobility
    }
  }
  phi_inv ~ exponential(5);
  
  // Likelihood
  for (p in 1:P) {
    cases[:, p] ~ neg_binomial_2(incidence[:, p], phi);
  }
}

generated quantities {
  array[P] real R0;              // Patch-specific R0
  array[T, P] real predicted_cases; // Posterior predictive
  array[T, P] real Rt;           // Effective reproduction number
  
  for (p in 1:P) {
    R0[p] = beta[p] / (gamma + mu);  // R0 including mortality
  }
  
  for (p in 1:P) {
    for (t in 1:T) {
      real S_p = y_pred[t, 1 + 6*(p-1)];  // Susceptibles at time t
      predicted_cases[t, p] = neg_binomial_2_rng(incidence[t, p], phi);
      Rt[t, p] = R0[p] * S_p / N[p];  // Rt = R0 * fraction susceptible
    }
  }
}