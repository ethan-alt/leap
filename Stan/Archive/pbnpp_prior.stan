
data {
  int<lower=0> n0;
  int<lower=0> p;
  vector[n0] y0;
  matrix[n0,p] X0;
  real<lower=0> prec_shape;
  real<lower=0> prec_rate;
  real<lower=0> a0_lower;
  real<lower=0> a0_upper;
  int<lower=1,upper=p> trtidx;
  int<lower=1,upper=p> covidx[p-1];
}
transformed data {
  int pm1 = p-1;
  matrix[n0,pm1] X0tilde = X0[1:n0, covidx];
  vector[pm1] betahat0 = mdivide_left_spd( crossprod(X0tilde), X0tilde' * y0 );
  real sse0 = dot_self(y0 - X0tilde * betahat0);
  real logC0 = -0.5 * pm1 * log(2*pi()) - 0.5 * log_determinant(crossprod(X0tilde));
  real delta0 = 2 * prec_shape;
  real gamma0 = 2 * prec_rate;
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
  real logC = -0.5 * pm1 * log_a0 + lgamma(shape_post) - shape_post * log(rate_post);
  target +=   (0.5 * (a0n0 + delta0) - 1) * log_prec 
            - 0.5 * a0 * prec * ( error0 + gamma0 * inv(a0) );
  target += -(logC + logC0);
  beta[trtidx] ~ normal(0, 10);
}

