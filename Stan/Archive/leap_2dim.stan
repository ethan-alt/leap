
data {
  int<lower=0> n;
  int<lower=0> n0;
  int<lower=0> p;
  vector[n] y;
  vector[n0] y0;
  matrix[n,p] X;
  matrix[n0,p] X0;
  real beta_mean;
  real<lower=0> beta_sd;
  real<lower=0> prec_shape;
  real<lower=0> prec_rate;
  real<lower=0> gamma_shape1;
  real<lower=0> gamma_shape2;
  real<lower=0,upper=1> gamma_lower;
  real<lower=gamma_lower,upper=1> gamma_upper;
}
parameters {
  vector[p] beta;
  vector[p] beta0;
  real<lower=0> prec;
  real<lower=0> prec0;
  real<lower=gamma_lower,upper=gamma_upper> gamma;
}
transformed parameters {
  real sigma = inv_sqrt(prec);   // std. dev
  real sigma0 = inv_sqrt(prec0);
  vector[n0] lp01;               // log probability of being in first component: log[ (gamma * N(mu1, sigma) / [gamma * N(mu1, sigma) + (1 -gamma) * N(mu0, sigma0)] ) ]
  vector[n0] loglik0;             // log mixture: log(gamma * N(mu1, sigma) + (1 - gamma) * N(mu0, sigma0))
  real log_gamma = log(gamma);
  real log1m_gamma = log1m(gamma);
  for ( i in 1:n0 ) {
    vector[2] contributions;
    contributions[1] = log_gamma + normal_lpdf(y0[i] | X0[i, ] * beta, sigma);
    contributions[2] = log1m_gamma + normal_lpdf(y0[i] | X0[i, ] * beta0, sigma0);
    loglik0[i] = log_sum_exp(contributions);
    lp01[i]    = contributions[1] - loglik0[i];
  }
  // for ( i in 1:n0 ) {
  //   lp01[i] = log_gamma
  //   lp0[i] = log_mix(
  //     gamma
  //     , normal_lpdf(y0[i] | mu01[i], sigma)
  //     , normal_lpdf(y0[i] | X0[i, ] * beta0, sigma0)
  //   );
  // }
}
model {
  // likelihood
  y ~ normal_id_glm(X, 0, beta, sigma);
  
  // LEAP
  target += loglik0;
  
  // initial priors
  if ( gamma_shape1 != 1 || gamma_shape2 != 1)
    gamma ~ beta(gamma_shape1, gamma_shape2);
  beta  ~ normal(beta_mean, beta_sd);
  beta0 ~ normal(beta_mean, beta_sd);
  prec  ~ gamma(prec_shape, prec_rate);
  prec0 ~ gamma(prec_shape, prec_rate);
}
generated quantities {
  real<lower=0> hist_ssc;
  int<lower=0,upper=1> c0[n0];
  for ( i in 1:n0 ) {
    c0[i] = bernoulli_rng(exp(lp01[i]));
  }
  hist_ssc = sum(c0);
}
