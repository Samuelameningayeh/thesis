functions {
  vector seir_patch(
    real t, 
    vector y, 
    array[] real theta, 
    array[] real x_r, 
    array[] int x_i
  ) {
    int P = x_i[1];               // Number of patches
    array[P] int N = x_i[2:P+1];  // Population sizes for each patch
    
    real sigma = theta[1];        // Incubation rate
    real gamma = theta[2];        // Recovery rate
    array[P] real beta = theta[3:2+P];  // Patch-specific transmission rates
    
    // Mobility matrix (flattened P x P)
    matrix[P, P] mobility;
    int idx = 3 + P;  // Start index for mobility parameters
    for (i in 1:P) {
      for (j in 1:P) {
        mobility[i, j] = theta[idx];
        idx += 1;
      }
    }
    
    vector[4 * P] dydt;  // Derivatives for S, E, I, R in all patches
    
    for (p in 1:P) {
      real S_p = y[1 + 4*(p-1)];
      real E_p = y[2 + 4*(p-1)];
      real I_p = y[3 + 4*(p-1)];
      real R_p = y[4 + 4*(p-1)];
      
      // Within-patch transmission
      real new_infections = beta[p] * S_p * I_p / N[p];
      
      // Cross-patch mobility effects
      real dS_in = 0.0;
      real dE_in = 0.0;
      real dI_in = 0.0;
      real dR_in = 0.0;
      real dS_out = 0.0;
      real dE_out = 0.0;
      real dI_out = 0.0;
      real dR_out = 0.0;
      
      for (k in 1:P) {
        if (k != p) {
          // Outflow from p to k
          dS_out += mobility[p, k] * S_p;
          dE_out += mobility[p, k] * E_p;
          dI_out += mobility[p, k] * I_p;
          dR_out += mobility[p, k] * R_p;
          
          // Inflow to p from k
          real S_k = y[1 + 4*(k-1)];
          real E_k = y[2 + 4*(k-1)];
          real I_k = y[3 + 4*(k-1)];
          real R_k = y[4 + 4*(k-1)];
          
          dS_in += mobility[k, p] * S_k;
          dE_in += mobility[k, p] * E_k;
          dI_in += mobility[k, p] * I_k;
          dR_in += mobility[k, p] * R_k;
        }
      }
      
      // ODEs for patch p
      dydt[1 + 4*(p-1)] = -new_infections - dS_out + dS_in;  // dS/dt
      dydt[2 + 4*(p-1)] = new_infections - sigma * E_p - dE_out + dE_in;  // dE/dt
      dydt[3 + 4*(p-1)] = sigma * E_p - gamma * I_p - dI_out + dI_in;  // dI/dt
      dydt[4 + 4*(p-1)] = gamma * I_p - dR_out + dR_in;  // dR/dt
    }
    
    return dydt;
  }
}

data {
  int<lower=1> P;               // Number of patches (3)
  int<lower=1> T;               // Time points
  real t0;                      // Initial time
  array[T] real ts;             // Observation times
  array[T, P] int<lower=0> cases; // Observed incidence
  array[P] int<lower=1> N;      // Population sizes
  array[P] real<lower=0> E0;    // Initial exposed
  array[P] real<lower=0> I0;    // Initial infected
}

transformed data {
  array[4 * P] real y0;         // Initial state (S1, E1, I1, R1, ..., SP, EP, IP, RP)
  array[0] real x_r;
  array[P + 1] int x_i;         // x_i[1] = P, x_i[2:P+1] = N[1:P]
  
  x_i[1] = P;
  for (p in 1:P) {
    x_i[p + 1] = N[p];
    y0[1 + 4*(p-1)] = N[p] - E0[p] - I0[p];  // S_p(0)
    y0[2 + 4*(p-1)] = E0[p];                 // E_p(0)
    y0[3 + 4*(p-1)] = I0[p];                 // I_p(0)
    y0[4 + 4*(p-1)] = 0;                     // R_p(0)
  }
}

parameters {
  real<lower=0> sigma;          // Incubation rate (1/latent period)
  real<lower=0> gamma;          // Recovery rate
  array[P] real<lower=0> beta;  // Patch-specific transmission rates
  array[P, P] real<lower=0> mobility;  // Mobility matrix (asymmetric)
  real<lower=0> phi_inv;        // Inverse dispersion
  array[P] real<lower=0, upper=1> rho;  // Reporting rates
}

transformed parameters {
  array[T, P] real incidence;
  real<lower=0> phi = 1.0 / phi_inv;
  array[3 + P + P*P] real theta;  // theta = [sigma, gamma, beta[1:P], mobility[P, P]]
  
  theta[1] = sigma;
  theta[2] = gamma;
  for (p in 1:P) {
    theta[2 + p] = beta[p];
  }
  for (i in 1:P) {
    for (j in 1:P) {
      theta[3 + P + (i-1)*P + j] = mobility[i, j];
    }
  }
  
  // Solve ODE
  array[T] vector[4 * P] y_pred = ode_rk45(
    seir_patch, 
    to_vector(y0), 
    t0, 
    ts, 
    theta, 
    x_r, 
    x_i
  );
  
  // Compute incidence per patch
  for (p in 1:P) {
    for (t in 2:T) {
        incidence[t, p] = y_pred[t, 3 + 4 * (p - 1)] - y_pred[t - 1, 3 + 4 * (p - 1)];
    }
    for (t in 1:T) {
        incidence[t, p] *= rho[p];  // Apply reporting rate element-wise
    }
  }
}

model {
  // Priors
  sigma ~ lognormal(log(1.0/5), 0.2);  // Ebola latent period ~5 days
  gamma ~ lognormal(log(1.0/10), 0.3); // Infectious period ~10 days
  for (p in 1:P) {
    beta[p] ~ lognormal(log(0.3), 0.2);
    rho[p] ~ beta(2, 2);
  }
  for (i in 1:P) {
    for (j in 1:P) {
      mobility[i, j] ~ exponential(0.1);  // Sparse mobility
    }
  }
  phi_inv ~ exponential(5);
  
  // Likelihood
  for (p in 1:P) {
    cases[:, p] ~ neg_binomial_2(incidence[:, p], phi);
  }
}

generated quantities {
  array[P] real R0; // Patch-specific R0
  array[T, P] real predicted_cases; // Posterior predictive for observed time points

  for (p in 1:P) {
    R0[p] = beta[p] / gamma;           // R0 = beta / gamma
  }
  
  // --- Posterior predictive for observed data ---
  for (p in 1:P) {
    for (t in 1:T) {
      predicted_cases[t, p] = neg_binomial_2_rng(incidence[t, p], phi);
    }
  }
}