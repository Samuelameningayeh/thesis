functions {
  vector seir(real t,
             vector y, 
             real beta, 
             real sigma,
             real gamma,
             real N) { 

      vector[5] dydt;

      real S = y[1];
      real E = y[2];
      real I = y[3];
      real R = y[4];
      
      dydt[1] = -beta * I * S / N;
      dydt[2] = beta * I * S / N - gamma * E;
      dydt[3] =  gamma * E - sigma * I;
      dydt[4] =  sigma * I;
      dydt[5] = gamma * E;
      
      return dydt;
  }
}
data {
  int<lower=1> n_days;
  vector[5] y0;
  real t0;
  array[n_days] real t;
  int N;
  array[n_days] int<lower=0> cases;
}

parameters {
  real<lower=0> sigma;
  real<lower=0> beta;
  real<lower=0> gamma;
  real<lower=0> phi_inv;
  //real<lower=0,upper=1> reporting_rate;
}
transformed parameters{
  real<lower=0,upper=1> reporting_rate;
  array[n_days] vector[5] y;
  vector[n_days] incidence;
  real<lower=0> phi = 1./phi_inv;
  
  y = ode_rk45(seir, y0, t0, t, beta, sigma, gamma, N);
  
  reporting_rate = 0.8;

  incidence[1] = y[1, 5] - 0;
  for (i in 2:n_days)
    incidence[i] = y[i, 5] - y[i-1, 5];
}
model {
    //priors
    beta ~ lognormal(log(0.4), 0.5);      // Infection rate prior
    sigma ~ lognormal(1.0 / 10, 0.5);   // Prior mean: 10-day incubation period
    gamma ~ lognormal(1.0/7, 0.5);   // Prior mean: 7-day infectious period 
    phi_inv ~ exponential(2);
    //reporting_rate ~ beta(2, 2);

    //sampling distribution
    cases ~ neg_binomial_2(reporting_rate*incidence+0.00001, phi);
}
generated quantities {
  real R0 = beta / sigma;
  real recovery_time = 1 / sigma;
  real incubation_period = 1 / gamma;

  array[n_days] real pred_incidence;
  pred_incidence = neg_binomial_2_rng(reporting_rate*incidence+0.00001, phi);
}
