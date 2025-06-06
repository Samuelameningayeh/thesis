functions {
  vector coupled_seir(real t, vector y_flat, 
                      array[] real N, 
                      array[] real beta, 
                      array[] real sigma, 
                      array[] real gamma,
                      array[] real kappa, 
                      array[] real alpha,
                      array[] real g, 
                      array[,] real m, 
                      array[,] real r) {
    int n = size(N);
    array[n, n, 6] real dydt;
    vector[n * n * 6] dydt_flat;
    
    // Reshape y_flat into y[n,n,6]
    array[n, n, 6] real y;
    for (i in 1:n) {
      for (j in 1:n) {
        for (k in 1:6) {
          y[i,j,k] = y_flat[(i-1)*n*6 + (j-1)*6 + k];
        }
      }
    }
    
    // Compute total population in each patch j
    array[n] real N_j;
    for (j in 1:n) {
      N_j[j] = 0;
      for (i in 1:n) {
        N_j[j] += sum(y[i,j,1:4]);
      }
      if (N_j[j] < 1e-6) N_j[j] = 1e-6;
    }
    
    // Compute derivatives
    for (i in 1:n) {
      for (j in 1:n) {
        real foi = 0;
        for (k in 1:n) {
          foi += kappa[j] * beta[i] * y[i,j,1] * y[k,j,3] / N_j[j];
        }
        
        real new_infections = sigma[i] * y[i,j,2];
        dydt[i,j,1] = -foi + g[i] * m[j,i] * y[i,i,1] - r[i,j] * y[i,j,1];
        dydt[i,j,2] = foi - new_infections + g[i] * m[j,i] * y[i,i,2] - r[i,j] * y[i,j,2];
        dydt[i,j,3] = new_infections - (gamma[i] + alpha[i]) * y[i,j,3] + g[i] * m[j,i] * y[i,i,3] - r[i,j] * y[i,j,3];
        dydt[i,j,4] = gamma[i] * y[i,j,3] + g[i] * m[j,i] * y[i,i,4] - r[i,j] * y[i,j,4];
        dydt[i,j,5] = alpha[i] * y[i,j,3];
        dydt[i,j,6] = new_infections;
      }
    }
    
    // Flatten dydt
    for (i in 1:n) {
      for (j in 1:n) {
        for (k in 1:6) {
          dydt_flat[(i-1)*n*6 + (j-1)*6 + k] = dydt[i,j,k];
        }
      }
    }
    return dydt_flat;
  }
}

data {
  int<lower=1> n_patches;
  int<lower=1> n_weeks;
  array[n_patches, n_patches] vector[6] y0;
  real t0;
  array[n_weeks] real ts;
  array[n_patches] real N;
  array[n_patches, n_weeks] int<lower=0> cases;
  array[n_patches] real beta_values;
  array[n_patches] real sigma_values;
  array[n_patches] real gamma_values;
  array[n_patches] real alpha_values;
  array[n_patches] real kappa_values;
  array[n_patches] real g_values;
  array[n_patches, n_patches] real m_values;
  array[n_patches, n_patches] real r_values;
  real<lower=0, upper=1> reporting_rate;
}

parameters {
  array[n_patches] real<lower=0> beta;
  array[n_patches] real<lower=0> sigma;
  array[n_patches] real<lower=0> gamma;
  array[n_patches] real<lower=0> alpha;
  real<lower=0> phi_inv;
}

transformed parameters {
  array[n_patches, n_patches, n_weeks] vector[6] y;
  array[n_patches, n_weeks] real weekly_incidence;
  real<lower=0> phi = 1.0 / phi_inv;
  
  // Flatten y0
  vector[n_patches * n_patches * 6] y0_flat;
  for (i in 1:n_patches) {
    for (j in 1:n_patches) {
      for (k in 1:6) {
        y0_flat[(i-1)*n_patches*6 + (j-1)*6 + k] = y0[i,j][k];
      }
    }
  }
  
  // Solve ODE
  array[n_weeks] vector[n_patches * n_patches * 6] y_flat = ode_rk45(coupled_seir, y0_flat, t0, ts, N, beta, sigma, gamma, alpha
                                                               kappa_values, g_values, m_values, r_values);
  
  // Reshape y_flat
  for (w in 1:n_weeks) {
    for (i in 1:n_patches) {
      for (j in 1:n_patches) {
        for (k in 1:6) {
          y[i,j,w][k] = y_flat[w][(i-1)*n_patches*6 + (j-1)*6 + k];
        }
      }
    }
  }
  
  // Compute incidence
  for (j in 1:n_patches) {
    for (w in 1:n_weeks) {
      real total_new = 0;
      for (i in 1:n_patches) {
        total_new += (w == 1) ? y[i,j,w][6] : (y[i,j,w][6] - y[i,j,w-1][6]);
      }
      weekly_incidence[j,w] = fmax(reporting_rate * total_new, 1e-6);
    }
  }
}

model {

   // PRIORS 
  for (p in 1:n_patches) {
    beta[p] ~ lognormal(log(beta_values[p]), 0.5);
    sigma[p] ~ lognormal(log(sigma_values[p]), 0.5);
    gamma[p] ~ lognormal(log(gamma_values[p]), 0.5);
    alpha[p] ~ lognormal(log(alpha_values[p]), 0.5);
  }
  phi_inv ~ exponential(5);
  
  // LIKELIHOOD
  for (j in 1:n_patches) {
      cases[j] ~ neg_binomial_2(weekly_incidence[j], phi);
  }
}

generated quantities {
  array[n_patches] real R0;
  array[n_patches] real incubation_period;
  array[n_patches] real recovery_period;
  array[n_patches, n_weeks] real pred_cases;
  
  for (p in 1:n_patches) {
    R0[p] = beta[p] * kappa_values[p] / fmax((gamma[p]+alpha[p]), 1e-6);
    incubation_period[p] = 1.0 / fmax(sigma[p], 1e-6);
    recovery_period[p] = 1.0 / fmax(gamma[p], 1e-6);
  }
  
  for (j in 1:n_patches) {
      pred_cases[j] = neg_binomial_2_rng(weekly_incidence[j], phi);
  }
}