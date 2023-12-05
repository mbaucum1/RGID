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
