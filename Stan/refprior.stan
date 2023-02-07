
data {
  int<lower=0> n;
  int<lower=0> p;
  vector[n] y;
  matrix[n,p] X;
  real beta_mean;
  real<lower=0> beta_sd;
  real sigma_mean;
  real<lower=0> sigma_sd;
  int<lower=0,upper=1> compute_loglik;
}
parameters {
  vector[p] beta;
  real<lower=0> sigma;
}
model {
  // likelihood
  y ~ normal_id_glm(X, 0, beta, sigma);
  // Initial prior
  beta ~ normal(beta_mean, beta_sd);
  sigma ~ normal(sigma_mean, sigma_sd);
}
generated quantities {
  vector[compute_loglik ? n : 0] log_lik;
  if (compute_loglik) {
    for ( i in 1:n ) {
      log_lik[i] = normal_lpdf(y[i] | X[i, ] * beta, sigma);
    }
  }
}
