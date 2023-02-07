
data {
  int<lower=0> n;          // current data sample size
  int<lower=0> n0;         // historical data sample size
  int<lower=0> p;          // number of covariates (incl. intercept)
  vector[n] y;             // current data response vector
  vector[n0] y0;           // historical data respnse vector
  matrix[n,p] X;           // current data design matrix (incl. intercept term)
  matrix[n0,p] X0;         // historical data design matrix (incl. intercept term)
  real beta_mean;          // mean for initial prior on reg coeffs
  real<lower=0> beta_sd;   // SD parameter for normal initial prior on reg coeffs
  real sigma_mean;         // mean parameter for half-normal prior on std. dev.
  real<lower=0> sigma_sd;  // SD parameter for half-normal initial prior on std. dev
  int<lower=0,upper=1> compute_ssc;
  int<lower=0,upper=1> compute_loglik;
}
transformed data {
  matrix[n0,p-1] X0tilde;
  array[p-1] int histindx;
  histindx[1] = 1;
  for ( i in 3:p ) 
    histindx[i-1] = i;
  X0tilde = X0[, histindx];
}
parameters {
  matrix[p, 2] betaMat;          // px2 matrix of regression coefficients; p = number of covars, 2 = number of components
  vector<lower=0>[2] sigma;      // 2-dim vector of std. devs
  real<lower=0,upper=1> gamma;
}
transformed parameters {
  vector[n0] lp01;
  vector[n0] loglik0;
  for ( i in 1:n0 ) {
    lp01[i] = normal_lpdf(y0[i] | X0tilde[i, ] * betaMat[histindx, 1], sigma[1]);
    loglik0[i] = log_mix(
      gamma
      , lp01[i]
      , normal_lpdf(y0[i] | X0tilde[i, ] * betaMat[histindx, 2], sigma[2])
    );
  }
}
model {
  // likelihood
  y ~ normal_id_glm(X, 0, betaMat[,1], sigma[1]);
    
  // LEAP is proportional to mixture density
  target += loglik0;
  
  // initial priors
  for ( k in 1:2 ) {
    betaMat[, k] ~ normal(beta_mean, beta_sd);
    sigma[k] ~ normal(sigma_mean, sigma_sd);
  }
}
generated quantities {
  vector[(compute_loglik) ? (n) : 0] log_lik;
  vector[(compute_ssc) ? (1) : (0)] hist_ssc;
  vector[(compute_ssc) ? (n0) : (0)] c01;
  vector[p] beta = betaMat[, 1];
  if ( compute_ssc ) {
    for ( i in 1:n0 )
      c01[i] = bernoulli_rng( exp( log(gamma) + lp01[i] - loglik0[i] ) );
    hist_ssc[1] = sum(c01); 
  }
  if ( compute_loglik )  {
    for ( i in 1:n )
      log_lik[i] = normal_lpdf(y[i] | X[i, ] * beta, sigma[1]);
  }
}
