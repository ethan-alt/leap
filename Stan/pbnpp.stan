
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
  real<lower=0> a0_shape1;
  real<lower=0> a0_shape2;
  real<lower=0,upper=1> a0_lower;
  real<lower=a0_lower,upper=1> a0_upper;
  int covidx[p-1];
  int trtidx;
  int<lower=0> compute_loglik;
  int<lower=0> compute_ssc;
}
transformed data {
  int pm1 = p-1;
  matrix[n0,pm1] X0tilde = X0[1:n0, covidx];
  vector[pm1] betahat0 = mdivide_left_spd( crossprod(X0tilde), X0tilde' * y0 );
  real sse0 = dot_self(y0 - X0tilde * betahat0);
  real logC0 = -0.5 * pm1 * log(2*pi()) - 0.5 * log_determinant(crossprod(X0tilde));
  real delta0 = 2 * prec_shape;
  real gamma0 = 2 * prec_rate;
  int not_unif = (a0_shape1 != 1 || a0_shape2 != 1) ? 1 : 0;
}
parameters {
  vector[p] beta;
  real<lower=0> prec;
  real<lower=a0_lower,upper=a0_upper> a0;
}
transformed parameters {
  real sigma = inv_sqrt(prec);
}
model {
  real error0 = dot_self(y0 - X0tilde * beta[covidx]);
  real a0n0 = a0 * n0;
  real log_prec = log(prec);
  real shape_post = 0.5 * (a0n0 + delta0 - pm1);
  real rate_post = 0.5 * (a0*sse0 + gamma0);
  real log_a0 = log(a0);
  // Normalizing constant for power prior | a0
  real logC = -0.5 * pm1 * log_a0 + lgamma(shape_post) - shape_post * log(rate_post);
  target += -(logC + logC0);
  
  // Kernel for power prior | a0
  target +=   (0.5 * (a0n0 + delta0) - 1) * log_prec 
            - 0.5 * a0 * prec * ( error0 + gamma0 * inv(a0) );
  // Prior for partial borrowing
  beta[trtidx] ~ normal(beta_mean, beta_sd);
  
  // Prior for a0
  if ( not_unif )
    a0 ~ beta(a0_shape1, a0_shape2);
  
  // Likelihood
  y ~ normal_id_glm(X, 0, beta, sigma);
}
generated quantities {
  vector[compute_ssc ? 1 : 0] hist_ssc;
  vector[compute_loglik ? n : 0] log_lik;
  if ( compute_ssc )
    hist_ssc[1] = a0 * n0;
  if ( compute_loglik ) {
    for ( i in 1:n )
      log_lik[i] = normal_lpdf(y[i] | X[i, ] * beta, sigma);
  }
}

