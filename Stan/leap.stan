
data {
  int<lower=0> n;          // current data sample size
  int<lower=0> n0;         // historical data sample size
  int<lower=0> p;          // number of covariates (incl. intercept)
  int<lower=0> K;          // number of components in mixture
  vector[n] y;             // current data response vector
  vector[n0] y0;           // historical data respnse vector
  matrix[n,p] X;           // current data design matrix (incl. intercept term)
  matrix[n0,p] X0;         // historical data design matrix (incl. intercept term)
  real beta_mean;          // mean for initial prior on reg coeffs
  real<lower=0> beta_sd;   // SD parameter for normal initial prior on reg coeffs
  real sigma_mean;         // mean parameter for half-normal prior on std. dev.
  real<lower=0> sigma_sd;  // SD parameter for half-normal initial prior on std. dev
  vector<lower=0>[K] conc; // Concentration parameter for exchangeability
  real<lower=0,upper=1> gamma_lower;            // Lower bound for probability of being exchangeable. 
  real<lower=gamma_lower,upper=1> gamma_upper;  // Upper bound for probability of being exchangeable.
  int<lower=0,upper=1> compute_ssc;
  int<lower=0,upper=1> compute_loglik;
}
transformed data {
  real gamma_shape1 = conc[1];
  real gamma_shape2 = sum(conc[2:K]);
  int K_gt_2 = (K > 2) ? (1) : (0);
  vector[K-1] conc_delta = conc[2:K];
}
parameters {
  matrix[p, K] betaMat;       // pxK matrix of regression coefficients; p = number of covars, K = number of components
  vector<lower=0>[K] sigma;  // K-dim vector of std. devs
  real<lower=gamma_lower,upper=gamma_upper> gamma;  // probability of being exchangeable
  simplex[K-1] delta_raw;
}
transformed parameters {
  vector[n0] lp01;                           // log probability for first component
  vector[n0] contrib;                        // log probability summing over all components
  vector[K] probs;
  vector[K] logProbs;
  probs[1]   = gamma;
  probs[2:K] = (1 - gamma) * delta_raw;
  logProbs = log(probs);
  // Compute probability of being in first component and marginalized log probability
  for ( i in 1:n0 ) {
    vector[K] contrib_i;   // K-dim vector giving log contribution for each component of historical data; total log contribution is log(sum(exp(.)))
    row_vector[K] mean0i = X0[i, ] * betaMat;
    for ( k in 1:K )
      contrib_i[k] = logProbs[k] + normal_lpdf(y0[i] | mean0i[k], sigma[k]);
    contrib[i]  = log_sum_exp(contrib_i);  // compute likelihood contribution of historical data
    lp01[i] = contrib_i[1] - contrib[i];  // compute log probability of being classified to first group
  }
}
model {
  // likelihood
  y ~ normal_id_glm(X, 0, betaMat[,1], sigma[1]);
    
  // LEAP is proportional to mixture density
  target += contrib;
  
  // initial priors
  for ( k in 1:K ) {
    betaMat[, k] ~ normal(beta_mean, beta_sd);
    sigma[k] ~ normal(sigma_mean, sigma_sd);
  }
  // If two components, get beta prior on gamma;
  // If >2 components, get a dirichlet prior on raw delta
  if ( gamma_shape1 != 1 || gamma_shape2 != 1)
    gamma ~ beta(gamma_shape1, gamma_shape2);
  if (K_gt_2)
    delta_raw ~ dirichlet(conc_delta);
}
generated quantities {
  vector[(compute_loglik) ? (n) : 0] log_lik;
  vector[(compute_ssc) ? (1) : (0)] hist_ssc;
  vector[(compute_ssc) ? (n0) : (0)] c01;
  vector[p] beta = betaMat[, 1];
  if ( compute_ssc ) {
    for ( i in 1:n0 )
      c01[i] = bernoulli_rng(exp(lp01[i]));
    hist_ssc[1] = sum(c01); 
  }
  if ( compute_loglik )  {
    for ( i in 1:n )
      log_lik[i] = normal_lpdf(y[i] | X[i, ] * beta, sigma[1]);
  }
}
