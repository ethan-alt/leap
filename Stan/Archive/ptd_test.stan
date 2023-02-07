
data {
  int K;
  real<lower=0,upper=1> lower;
  real<lower=lower,upper=1> upper;
  vector<lower=0>[K] conc;
}
transformed data {
  real shape1 = conc[1];
  real shape2 = sum(conc[2:K]);
  vector[K-1] conc_x2 = conc[2:K];
  int K_gt_2 = (K > 2) ? 1 : 0;
}
// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real<lower=lower,upper=upper> x1;
  simplex[K-1] x2_raw;
}
transformed parameters {
  vector[K] x;
  x[1]   = x1;
  x[2:K] = (1 - x1) * x2_raw;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  if (shape1 != 1 || shape2 != 1)
    x1 ~ beta(shape1, shape2);
  if (K_gt_2)
    x2_raw ~ dirichlet(conc_x2);
}

