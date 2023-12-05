#Dependencies
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels.api as sm

#Experimental functions
#Function for generating dataset with k features, the first k_cor of which are correlated at r
def generate_data(k = 6, cor_indices = None, n = 100000, r=0.5, seed = 2023):
  mean_vec = np.zeros(k)
  cov_matrix = np.diag(np.zeros(k))
  for c1 in cor_indices:
    for c2 in cor_indices:
      cov_matrix[c1,c2] = r

  for d in range(k):
    cov_matrix[d,d] = 1

  np.random.seed(seed)

  X = np.random.multivariate_normal(mean_vec, cov_matrix, size=n)
  return X


#Create response function
def generate_response(data, linear_coefs, nonlinear_coefs, log_coefs, stepwise_coefs, interaction_tuples,
              interaction_types, int_coefs = 1, nonlinear_exp = 2, stepwise_cutoff = 0, noise_sd = 0,
                      log_adjust = 3, seed = 32301):

  k = data.shape[1]
  #Pad with zeros for right-appended nuisance paramters
  if len(linear_coefs) < k:
    linear_coefs = np.concatenate([linear_coefs, np.zeros(k - len(linear_coefs))])
  if len(nonlinear_coefs) < k:
    nonlinear_coefs = np.concatenate([nonlinear_coefs, np.zeros(k - len(nonlinear_coefs))])
  if len(stepwise_coefs) < k:
    stepwise_coefs = np.concatenate([stepwise_coefs, np.zeros(k - len(stepwise_coefs))])
  if len(log_coefs) < k:
    log_coefs = np.concatenate([log_coefs, np.zeros(k - len(log_coefs))])

  y_linear = (np.array(linear_coefs).reshape(1, -1) * data).sum(axis = 1)
  y_nonlinear = (np.array(nonlinear_coefs).reshape(1, -1) * (data**nonlinear_exp)).sum(axis = 1)
  y_stepwise = (np.array(stepwise_coefs).reshape(1, -1) * (data > stepwise_cutoff)).sum(axis = 1)
  y_log = (np.array(log_coefs).reshape(1, -1) * (np.log(np.clip(data + log_adjust, a_min = .001, a_max = None)))).sum(axis = 1)

  if len(int_coefs)==1:
    int_coefs = [int_coefs]*len(interaction_tuples)
  y_int = np.zeros(data.shape[0])
  for t, c, interaction_type in zip(interaction_tuples, int_coefs, interaction_types):
    if interaction_type == 'linear':
      y_int += c * np.prod(data[:, t], axis = 1)
    if interaction_type == 'nonlinear':
      y_int += c * (data[:,t[0]]**nonlinear_exp) * (data[:,t[1]]**nonlinear_exp)
    if interaction_type == 'stepwise':
      y_int += c * data[:,t[0]] * np.prod( (data[:,t[1:]] > stepwise_cutoff), axis = 1)
    if interaction_type == 'log': #note: Subtracting 1 to effectively mean-center the log-functions
      y_int += c * data[:,t[0]] * np.prod(np.log(np.clip(data[:,t[1:]] + log_adjust,a_min =.001,a_max=None)) - 1, axis = 1)

  ybar = y_linear + y_nonlinear + y_stepwise + y_log + y_int
  print(f'Linear, nonlinear, stepwise, and log SD are {np.std(y_linear)},{np.std(y_nonlinear)},{np.std(y_stepwise)},{np.std(y_log)}')
  np.random.seed(seed)
  yerr = np.random.normal(loc = 0, scale =noise_sd, size = ybar.shape[0])
  y = ybar + yerr
  print('ybar SD is '+ str(np.std(ybar)))
  print('y SD is '+str(np.std(y)))
  print('R^2 is '+str(np.sum(ybar**2) / np.sum(y**2)))

  return y

#Function for splitting data
def train_val_test_split(X, y, train_prop = 0.6, val_prop = 0.2):
  train_size = int(np.round(train_prop * X.shape[0]))
  val_size = int(np.round(val_prop * X.shape[0]))

  if isinstance(X, pd.core.frame.DataFrame):
    X_train = X.iloc[0:train_size,:]
    y_train = y.iloc[0:train_size]

    X_val = X.iloc[train_size:(train_size+val_size),:]
    y_val = y.iloc[train_size:(train_size+val_size)]

    X_test = X.iloc[(train_size+val_size):,:]
    y_test = y.iloc[(train_size+val_size):]
  else:
    X_train = X[0:train_size,:]
    y_train = y[0:train_size]

    X_val = X[train_size:(train_size+val_size),:]
    y_val = y[train_size:(train_size+val_size)]

    X_test = X[(train_size+val_size):,:]
    y_test = y[(train_size+val_size):]

  return X_train, y_train, X_val, y_val, X_test, y_test

def train_test_split(X, y, train_prop = 0.8):
  train_size = int(np.round(train_prop * X.shape[0]))
  if isinstance(X, pd.core.frame.DataFrame):
    X_train = X.iloc[0:train_size,:]
    y_train = y.iloc[0:train_size]
    X_test = X.iloc[train_size:,:]
    y_test = y.iloc[train_size:]
  else:
    X_train = X[0:train_size,:]
    y_train = y[0:train_size]
    X_test = X[train_size:,:]
    y_test = y[train_size:]

  return X_train, y_train, X_test, y_test

#Functions for benchmark computational experiments

def hsquared(model, f, X, grid_size = 20, predict_proba = False):
  X = np.array(X)
  points = cutpoints(X[:,f], grid_size)['values'].values
  print(points)
  grid_size = len(points)
  preds_x = model.predict(X) if predict_proba == False else model.predict_proba(X)
  #exp_dataset = np.tile(X, (X.shape[0],1))
  exp_dataset = np.tile(X, (grid_size,1))
  exp_dataset[:,f] = np.repeat(points, X.shape[0])
  if len(np.repeat(points, X.shape[0])) != exp_dataset.shape[0]:
    print('mismatch in imputed dataset and imputed feature')
  # using predict.pre for a huge dataset may lead to errors, so split computations up in k parts:
  k = 10
  ids = np.random.choice(np.arange(k), exp_dataset.shape[0], replace = True)
  yhat = np.zeros(exp_dataset.shape[0])
  for i in np.arange(k):
    yhat[ids==i] = model.predict(exp_dataset[ids==i,:]) if predict_proba == False else model.predict_proba(exp_dataset[ids==i,:])
  i_xj = np.repeat(np.arange(grid_size), X.shape[0])
  preds_xj = pd.Series(yhat).groupby(i_xj).mean() #preds_xj is a series, with the average predicted value for each xj
  preds_xj = np.interp(X[:,f], points, preds_xj.to_numpy())

  i_xnotj = np.tile( np.arange(X.shape[0]) , grid_size)
  preds_xnotj = pd.Series(yhat).groupby(i_xnotj).mean()
  # H should be calculated based on centered functions:
  preds_x = preds_x - preds_x.mean()
  preds_xj = (preds_xj - preds_xj.mean()) #.to_numpy()
  preds_xnotj = (preds_xnotj - preds_xnotj.mean()).to_numpy()
  return np.sum( (preds_x - preds_xj - preds_xnotj)**2) / np.sum(preds_x**2)

def generate_random_data(n_t, _lambda, m, n, eigen_list, sn_ratio, seed):
  eigen_a_c, eigen_b_c, eigen_a_g, eigen_b_g = eigen_list

  #Set seed
  np.random.seed(seed)

  #Generate original features
  mean_vec = np.zeros(m)
  U_c = scipy.stats.ortho_group.rvs(dim=m)
  D_c = np.diag(np.random.uniform(eigen_a_c, eigen_b_c, m)**2)
  cov_matrix = U_c @ D_c @ U_c.T
  X = np.random.multivariate_normal(mean_vec, cov_matrix, size = n)
  for i in range(X.shape[1]):
    X[:,i] = (X[:,i] - X[:,i].mean()) / np.std(X[:,i])
  avg_cor = (np.abs(pd.DataFrame(X).corr()).sum().sum() - m) / (m**2 - m)
  print('Average correlation is '+str(avg_cor))

  #Generate target
  a_l = np.random.uniform(-1, 1, n_t) #term coefficients
  n_l = np.clip(
      np.floor(np.random.exponential(_lambda, n_t) + 1.5).astype('int'),
      a_min = None, a_max = m) #no. features in each term
  z_l = [np.random.choice(range(m), _n_l, replace = False) for _n_l in n_l] #list of feature groups for each term
  y = np.zeros((n,1))
  for t in range(n_t):
    mean_vec_l = np.random.normal(size = n_l[t]) #using a standard normal matrix for the mean vec (as in Friedman), though note that the original data matrix has covariance (and this one doesnt)
    if n_l[t] > 1:
      U_g = scipy.stats.ortho_group.rvs(dim=n_l[t])
    else:
      U_g = np.diag([1])
    D_g = np.diag(np.random.uniform(eigen_a_g, eigen_b_g, n_l[t])**2)
    cov_matrix_l = U_g @ D_g @ U_g.T

    z = X[:,z_l[t]]
    g = np.exp(-0.5*(((z - mean_vec_l.reshape(1,-1)) @ cov_matrix_l) * (z - mean_vec_l.reshape(1,-1))).sum(axis = 1))
    g = g.reshape(-1,1)

    y += a_l[t] * np.array(g)

  y = y.reshape(-1)
  #Now scale error according to s/n ratio
  MAD = np.mean(np.abs(y - np.median(y))) #this is from Friedman 2001
  error_sd = (1/sn_ratio * MAD) / 0.8 #because the MAD is about 0.8 times the MAD.
  yerr = np.random.normal(loc = 0, scale = error_sd, size = len(y))
  y_final = y + yerr
  print('Ybar median abs. deviation is '+str(MAD))
  print('Error median abs. deviation is '+str(np.mean(np.abs(yerr - np.median(yerr)))))

  return X, y_final, z_l, a_l

