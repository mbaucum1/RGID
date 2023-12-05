#These functions moved to Github on 12/4/2023
def neg_auc(y_true, y_pred):
  return -1 * roc_auc_score(y_true, y_pred)
def rmse(y_true, y_pred):
  return np.sqrt(np.mean((y_true - y_pred)**2))
def mse(y_true, y_pred):
  return np.mean((y_true - y_pred)**2)
def mae(y_true, y_pred):
  return np.mean(np.abs(y_true - y_pred))

def retrieve_quantiles(w):
  return np.cumsum(np.array([0]+list(w))/2 + np.array(list(w)+[0]) / 2)[0:-1]

def threeway_h2(f1_ice, f2_ice, f3_ice, f1_filters, f2_filters, f3_filters,
                f1_weights, f2_weights, f3_weights, grid_size,
               idf, f1, f2, f3, fnames, missing_threshold):

    max_missingness = 0

    grid_size1, grid_size2, grid_size3 = f1_ice.shape[0], f2_ice.shape[0], f3_ice.shape[0]
      #PDP version 1
    f1_zero_counts = np.any([(f2 & f3).sum() > 0 for f2 in f2_filters for f3 in f3_filters])
    if f1_zero_counts:
      pdp3_f1_f2f3 = np.concatenate(
      [np.array(
          [np.sum(f1_ice[:,f2 & f3] * ((w2*w3)[f2 & f3]).reshape(1,-1),axis=1) / np.max([np.sum((w2*w3)[f2 & f3]),1e-5]) for f2, w2 in zip(f2_filters, f2_weights)]).T.reshape(grid_size1,grid_size2,1) for f3,w3 in zip(f3_filters,f3_weights)], axis = 2)
      max_missingness = np.max([max_missingness, np.mean(pdp3_f1_f2f3==0)])
      pdp3_f1_f2f3[pdp3_f1_f2f3==0] = np.mean(pdp3_f1_f2f3[pdp3_f1_f2f3!=0])
    else:
      pdp3_f1_f2f3 = np.concatenate(
      [np.array(
        [np.average(f1_ice[:,f2 & f3], axis = 1, weights = (w2*w3)[f2 & f3]
                    ) for f2, w2 in zip(f2_filters, f2_weights)]).T.reshape(grid_size1,grid_size2,1) for f3,w3 in zip(f3_filters,f3_weights)], axis = 2)
      #PDP version 2
    f2_zero_counts = np.any([(f1 & f3).sum() > 0 for f1 in f1_filters for f3 in f3_filters])
    if f2_zero_counts:
      pdp3_f2_f1f3 = np.concatenate(
        [np.array(
            [np.sum(f2_ice[:,f1 & f3] * ((w1*w3)[f1&f3]).reshape(1,-1),axis = 1) / np.max([np.sum((w1*w3)[f1&f3]),1e-5]) for f1, w1 in zip(f1_filters,f1_weights)]).reshape(grid_size1,grid_size2,1) for f3, w3 in zip(f3_filters, f3_weights)],axis = 2)
      max_missingness = np.max([max_missingness, np.mean(pdp3_f2_f1f3==0)])
      pdp3_f2_f1f3[pdp3_f2_f1f3==0] = np.mean(pdp3_f2_f1f3[pdp3_f2_f1f3!=0])
    else:
      pdp3_f2_f1f3 = np.concatenate(
        [np.array(
            [np.average(f2_ice[:,f1 & f3], axis = 1, weights = (w1*w3)[f1 & f3]
                      ) for f1, w1 in zip(f1_filters,f1_weights)]).reshape(grid_size1,grid_size2,1) for f3, w3 in zip(f3_filters, f3_weights)],axis = 2)
      #PDP version 3
    f3_zero_counts = np.any([(f1 & f2).sum() > 0 for f1 in f1_filters for f2 in f2_filters])
    if f3_zero_counts:
      pdp3_f3_f1f2 = np.array(
        [np.array(
            [np.sum(f3_ice[:, f1 & f2] * ((w1*w2)[f1&f2]).reshape(1,-1), axis = 1)/np.max([np.sum((w1*w2)[f1&f2]),1e-5]) for f2, w2 in zip(f2_filters, f2_weights)]
            ) for f1, w1 in zip(f1_filters, f1_weights)]
      )
      max_missingness = np.max([max_missingness, np.mean(pdp3_f3_f1f2==0)])
      pdp3_f3_f1f2[pdp3_f3_f1f2==0] = np.mean(pdp3_f3_f1f2[pdp3_f3_f1f2!=0])
    else:
      pdp3_f3_f1f2 = np.array(
        [np.array(
            [np.average(f3_ice[:, f1 & f2], axis = 1, weights = (w1*w2)[f1 & f2]) for f2, w2 in zip(f2_filters, f2_weights)]
            ) for f1, w1 in zip(f1_filters, f1_weights)]
      )

    if pdp3_f1_f2f3.shape != (grid_size1,grid_size2,grid_size3):
      print('PDP3 v1 has shape mismatch')
    if pdp3_f2_f1f3.shape != (grid_size1,grid_size2,grid_size3):
      print('PDP3 v1 has shape mismatch')
    if pdp3_f3_f1f2.shape != (grid_size1,grid_size2,grid_size3):
      print('PDP3 v1 has shape mismatch')

            #Average the PDPs
    pdp3_final = np.average(
        np.concatenate([np.expand_dims(pdp3_f1_f2f3,0),np.expand_dims(pdp3_f2_f1f3,0),np.expand_dims(pdp3_f3_f1f2,0)],axis=0
        ), axis = 0)
        #weights = np.concatenate([np.expand_dims(pdp3_f1_f2f3_COUNTS,0),np.expand_dims(pdp3_f2_f1f3_COUNTS,0),np.expand_dims(pdp3_f3_f1f2_COUNTS,0)],axis=0)

    pdp3_final = pdp3_final - pdp3_final.mean()
    pdp3_f2f3, pdp3_f1f3, pdp3_f1f2 = [np.expand_dims(np.mean(pdp3_final, axis = a),a) for a in [0,1,2]]
    pdp3_f1, pdp3_f2, pdp3_f3 = [np.expand_dims(np.mean(pdp3_final, axis = tup), axis = tup) for tup in [(1,2),(0,2),(0,1)]]

    full_h2 =  np.sum(
     (pdp3_final - pdp3_f1f2 - pdp3_f2f3 - pdp3_f1f3 + pdp3_f1 + pdp3_f2 + pdp3_f3)**2
    ) / np.sum(pdp3_final**2)

    cor12 = np.corrcoef(pdp3_f1_f2f3.reshape(-1),pdp3_f2_f1f3.reshape(-1))[0,1]
    cor23 = np.corrcoef(pdp3_f2_f1f3.reshape(-1),pdp3_f3_f1f2.reshape(-1))[0,1]
    cor13 = np.corrcoef(pdp3_f1_f2f3.reshape(-1),pdp3_f3_f1f2.reshape(-1))[0,1]
    avg_cor = np.mean([cor12,cor23,cor13])

    if (max_missingness < missing_threshold):
        #resid_h2 =  np.float(full_h2)
        pfull_h2_12= np.sum((pdp3_final.mean(axis = 2) - pdp3_final.mean(axis = 2).mean(axis=0).reshape(1,-1) - pdp3_final.mean(axis = 2).mean(axis=1).reshape(-1,1) )**2) / np.sum(pdp3_final.mean(axis = 2)**2)
        pfull_h2_23 = np.sum((pdp3_final.mean(axis = 0) - pdp3_final.mean(axis = 0).mean(axis=0).reshape(1,-1) - pdp3_final.mean(axis = 0).mean(axis=1).reshape(-1,1) )**2) / np.sum(pdp3_final.mean(axis = 0)**2)
        pfull_h2_13 = np.sum((pdp3_final.mean(axis = 1) - pdp3_final.mean(axis = 1).mean(axis=0).reshape(1,-1) - pdp3_final.mean(axis = 1).mean(axis=1).reshape(-1,1) )**2) / np.sum(pdp3_final.mean(axis = 1)**2)

        if fnames is None:
          real_h2_12 = idf[(idf['Feature'] == 'X'+str(f1)) & (idf['Feature 2'] == 'X'+str(f2))]['Diff'].squeeze()
          real_h2_23 = idf[(idf['Feature'] == 'X'+str(f2)) & (idf['Feature 2'] == 'X'+str(f3))]['Diff'].squeeze()
          real_h2_13 = idf[(idf['Feature'] == 'X'+str(f1)) & (idf['Feature 2'] == 'X'+str(f3))]['Diff'].squeeze()
        else:
          real_h2_12 = idf[(idf['Feature'] == fnames[f1]) & (idf['Feature 2'] == fnames[f2])]['Diff'].squeeze()
          real_h2_23 = idf[(idf['Feature'] == fnames[f2]) & (idf['Feature 2'] == fnames[f3])]['Diff'].squeeze()
          real_h2_13 = idf[(idf['Feature'] == fnames[f1]) & (idf['Feature 2'] == fnames[f3])]['Diff'].squeeze()

        pfull_vec = np.clip([pfull_h2_12, pfull_h2_23, pfull_h2_13], a_min = 0.00000001, a_max = None)
        avg_pdp_h2 = np.exp(np.mean(np.log(pfull_vec)))
        h2_vec = np.clip([real_h2_12, real_h2_23, real_h2_13], a_min = 0.00000001, a_max = None)
        avg_real_h2 = np.exp(np.mean(np.log(h2_vec)))

        resid_h2 = (full_h2 / avg_pdp_h2) * avg_real_h2

    else:
      resid_h2 = 0
    return resid_h2, avg_cor

def get_neighborhood_points(x, q_central, neighborhood_size):
  qmin = np.max([q_central - neighborhood_size/2, 0])
  qmax = np.min([q_central + neighborhood_size/2, 1])
  if np.isclose(np.quantile(x, qmin), np.quantile(x, qmax)):
    filt = (x == np.quantile(x, qmin))
  else:
    filt = (x > np.quantile(x, qmin)) & (x <= np.quantile(x, qmax))
  if np.sum(filt) == 0: #try including lower bound
    filt = (x >= np.quantile(x, qmin)) & (x <= np.quantile(x, qmax))

  if np.all(filt==True):
    idx = np.random.choice(np.arange(len(x)), size = int(np.floor(neighborhood_size * len(x))), replace = False)
    filt = np.array([False] * len(x))
    filt[idx] = True
  return filt

def propensity_filters_and_weights(f, pdp_w_list, pdp2_band_width, pdp3_band_width,X,pdp_ips_trim_q, propensity_samples = 1000):
  X = np.array(X)

  X_lr = (np.array(X) - np.array(X).mean(axis = 0).reshape(1,-1)) / np.array(X).std(axis=0).reshape(1,-1) #standardize so that LR expected value is always 50%. This reduces variance in weights
  if propensity_samples >= X_lr.shape[0]:
    X_lr_train = np.array(X_lr)
    sample_index = np.arange(X_lr.shape[0])
  else:
    np.random.seed(propensity_samples)
    sample_index = np.random.choice(range(X_lr.shape[0]), size = propensity_samples)
    X_lr_train = np.array(X_lr)[sample_index,:] #note that X_lr_train is standardized based on X_lr (which may be larger), but that's okay

  regularize = False

  if np.linalg.matrix_rank(X_lr_train) < X_lr_train.shape[1]:
    regularize = True

  f_quantiles = np.array(retrieve_quantiles(pdp_w_list[f]))

  f_quantile_bands2 = [(np.max([q - pdp2_band_width/2,0]), np.min([q + pdp2_band_width/2,1])) for q in f_quantiles]

  f_quantile_bands3 = [(np.max([q - pdp3_band_width/2,0]), np.min([q + pdp3_band_width/2,1])) for q in f_quantiles]

  #If variable has more values than weights (meaning more values than grid_size), use the standard formula
  unique = np.unique(X[:,f])
  if len(unique) > len(pdp_w_list[f]): #i.e., if there are enough values that cutpoints() shortened to grid_size
    f_filters2 = [ (X[:,f] > np.quantile(X[:,f], q1)) & (X[:,f] <= np.quantile(X[:,f], q2)) \
        for (q1, q2) in f_quantile_bands2 ]
    if np.any([f.sum()==0 for f in f_filters2]): #include lower bound if too few values
        f_filters2 = [ (X[:,f] >= np.quantile(X[:,f], q1)) & (X[:,f] <= np.quantile(X[:,f], q2)) \
        for (q1, q2) in f_quantile_bands2 ]

    f_filters3 = [(X[:,f] > np.quantile(X[:,f], q1)) & (X[:,f] <= np.quantile(X[:,f], q2)) \
        for (q1, q2) in f_quantile_bands3]
    if np.any([f.sum()==0 for f in f_filters3]): #include lower bound if too few values
        f_filters3 = [ (X[:,f] >= np.quantile(X[:,f], q1)) & (X[:,f] <= np.quantile(X[:,f], q2)) \
        for (q1, q2) in f_quantile_bands3 ]

  else:
    f_filters2 = [ X[:,f]==u for u in unique]
    f_filters3 = [ X[:,f]==u for u in unique]

  if (np.any([f.sum()==0 for f in f_filters2])) | (np.any([f.sum()==0 for f in f_filters3])):
    print(f)
    print(pdp_w_list[f])
    print(f_quantile_bands2)
    print(f_quantile_bands3)
  if regularize == True:
    f_propensities2 = [LogisticRegression(penalty='l2',fit_intercept=False).fit(
          np.delete(X_lr_train, f, axis = 1), filt[sample_index]).predict_proba(np.delete(X_lr, f, axis = 1))[:,1] for filt in f_filters2]
    f_propensities3 = [LogisticRegression(penalty='l2',fit_intercept=False).fit(
          np.delete(X_lr_train, f, axis = 1), filt[sample_index]).predict_proba(np.delete(X_lr, f, axis = 1))[:,1] for filt in f_filters3]
    
  else:
    if X_lr_train.shape[0] < X_lr.shape[0]: #if using different train and test matrices
      f_propensities2 = [ sm.Logit(filt[sample_index], np.delete(X_lr_train, f, axis = 1)).fit(disp=0).predict(np.delete(X_lr, f, axis = 1)) for filt in f_filters2]
      f_propensities3 = [sm.Logit(filt[sample_index], np.delete(X_lr_train, f, axis = 1)).fit(disp=0).predict(np.delete(X_lr, f, axis = 1)) for filt in f_filters3]
    else:
      f_propensities2 = [ sm.Logit(filt, np.delete(X_lr, f, axis = 1)).fit(disp=0).predict() for filt in f_filters2]
      f_propensities3 = [sm.Logit(filt, np.delete(X_lr, f, axis = 1)).fit(disp=0).predict() for filt in f_filters3]

  f_weights2 = [np.clip(1/np.clip(fp,a_max=None,a_min=1e-4), a_min = None, a_max = np.quantile(1/np.clip(fp,a_max=None,a_min=1e-4), pdp_ips_trim_q)) for fp in f_propensities2]
  f_weights3 = [np.clip(1/np.clip(fp,a_max=None,a_min=1e-4), a_min = None, a_max = np.quantile(1/np.clip(fp,a_max=None,a_min=1e-4), pdp_ips_trim_q)) for fp in f_propensities3]

  return f_filters2, f_weights2, f_filters3, f_weights3

def pdp_propensity_filters_and_weights(f, pdp_high_filter, pdp_low_filter, X, pdp_ips_trim_q, propensity_samples = 1000):
  #Computing 2PDQUIVER propensities outside their own function due to simplicity
  X_lr = (np.array(X) - np.array(X).mean(axis = 0).reshape(1,-1)) / np.array(X).std(axis=0).reshape(1,-1) #standardize so that LR expected value is always 50%. This reduces variance in weights

  if propensity_samples >= X_lr.shape[0]:
    X_lr_train = np.array(X_lr)
    sample_index = np.arange(X_lr.shape[0])
  else:
    np.random.seed(propensity_samples)
    sample_index = np.random.choice(range(X_lr.shape[0]), size = propensity_samples)
    X_lr_train = np.array(X_lr)[sample_index,:] #note that X_lr_train is standardized based on X_lr (which may be larger), but that's okay

  regularize = False
  if np.linalg.matrix_rank(X_lr_train) < X_lr_train.shape[1]:
    regularize = True

  if regularize == True: #fit from X_lr_train, predict based on X_lr (which may be same matrix if X.shape[0] < propensity_samples)
    high_propensities = LogisticRegression(penalty='l2',fit_intercept=False).fit(
        np.delete(X_lr_train, f, axis = 1),
        pdp_high_filter[sample_index,f] #use sample_index to subset y (which may just be all rows)
        ).predict_proba(np.delete(X_lr, f, axis = 1))[:,1]
    low_propensities = LogisticRegression(penalty='l2',fit_intercept=False).fit(
        np.delete(X_lr_train, f, axis = 1),
        pdp_low_filter[sample_index,f] #use sample_index to subset y (which may just be all rows)
        ).predict_proba(np.delete(X_lr, f, axis = 1))[:,1]
    
  else:
    if X_lr_train.shape[0] < X_lr.shape[0]: #if using different train matrix from X, fit from X_lr_train and predict from X_lr
      high_propensities = sm.Logit(pdp_high_filter[sample_index,f], np.delete(X_lr_train, f, axis = 1)).fit(disp=0).predict(np.delete(X_lr, f, axis = 1)) #get propensities for that entire column being above its qth quantile
      low_propensities = sm.Logit(pdp_low_filter[sample_index,f], np.delete(X_lr_train, f, axis = 1)).fit(disp=0).predict(np.delete(X_lr, f, axis = 1)) #get propensities for that entire column being below its qth quantile
    else: #if X_lr and X_lr_train are identical, just fit/predict from X_lr
      high_propensities = sm.Logit(pdp_high_filter[:,f], np.delete(X_lr, f, axis = 1)).fit(disp=0).predict() #get propensities for that entire column being above its qth quantile
      low_propensities = sm.Logit(pdp_low_filter[:,f], np.delete(X_lr, f, axis = 1)).fit(disp=0).predict() #get propensities for that entire column being below its qth quantile

  high_weights = np.clip(1/np.clip(high_propensities,a_max=None,a_min=1e-4), a_min=None, a_max = np.quantile(1/np.clip(high_propensities,a_max=None,a_min=1e-4), pdp_ips_trim_q)) #convert to weights
  low_weights = np.clip(1/np.clip(low_propensities,a_max=None,a_min=1e-4), a_min=None, a_max = np.quantile(1/np.clip(low_propensities,a_max=None,a_min=1e-4), pdp_ips_trim_q)) #convert to weights

  return high_weights, low_weights


 #Write and run function getting cutpoints for feature f
def cutpoints(var, grid_size): #cutpoints are MIDPOINTS of evenly divided segments
  if len(np.unique(var)) <= grid_size:
    values = np.sort(np.unique(var))
    weights = np.array([np.mean(var == v) for v in values])
    df_final = pd.DataFrame({'values': values, 'weights': weights})
  else:
    width = 1 / grid_size
    quantiles = np.arange(0, 1, width) + width/2
    weights = np.ones(grid_size) / grid_size
    values = np.array([np.quantile(var, q, method = 'inverted_cdf') for q in quantiles])
    values = np.round(values, 3) #rounding allows us to lump similar quantiles together
    df = pd.DataFrame({'values': values, 'weights': weights})
    df_final = df.groupby('values', as_index = False)['weights'].sum()
  return df_final


#Create feature-specific PDP function
def feature_importance_scores(X, y, model, f, metric, grid_size = 20, partial_out = True,
                              predict_proba = False, X_residuals = None, interaction_quantiles = (0.25, 0.75)):

  #Choose a continuous feature, then make predictions for all imputed values of that feature
  if isinstance(X, pd.core.frame.DataFrame):
    is_df = True
    fnames = X.columns
  else:
    is_df = False
  #Identify grid points for variable f
  c = cutpoints(X.iloc[:,f], grid_size = grid_size) if is_df == True else cutpoints(X[:,f], grid_size = grid_size)
  weights = c['weights'].values
  values = c['values'].values

  #Print number of unique values
  n_unique = len(np.unique(X.iloc[:,f])) if is_df == True else len(np.unique(X[:,f]))
  if n_unique < len(values):
    print('Number of quantile points greater than number of unique values')

  #Create two-way PDP matrix
  pdp = np.zeros((len(values), X.shape[0]))

  #Loop through all cutpoints values
  X_impute = X.copy(deep = True) if is_df == True else np.array(X)

  for v, i in zip(values, range(len(values))):
    if is_df == True:
     X_impute.iloc[:,f] = v
    else:
     X_impute[:,f] = v
    _yhat = model.predict(X = X_impute) if predict_proba == False else model.predict_proba(X= X_impute)[:,1] #get '1' class probabilities

    pdp[i, :] = _yhat.reshape(-1)
    #pdp[i, :] = _yhat_results[i].reshape(-1)

    #Create one-way PDP's by averaging across both axes
  #if conditional==False:
  f_not_pdp = np.average(pdp, axis = 0, weights = weights)
  f_pdp = pdp.mean(axis=1).reshape(-1)
  f_pdp_interp = np.interp(X.iloc[:,f], values, f_pdp) if is_df == True else np.interp(X[:,f], values, f_pdp)

    #Center both pdps
  f_not_pdp = f_not_pdp - f_not_pdp.mean()
  f_pdp_interp = f_pdp_interp - f_pdp_interp.mean()

  #Add linear trendline to PDP
  if is_df==True:
    lm_model = sm.OLS(f_pdp_interp.reshape(-1), sm.add_constant(X.iloc[:,f]))
    results = lm_model.fit()
    linear_pdp_component = results.params[0] + results.params[1] * X.iloc[:,f] #has n entries

  else:
    lm_model = sm.OLS(f_pdp_interp.reshape(-1), sm.add_constant(X[:,f]))
    results = lm_model.fit()
    linear_pdp_component = results.params[0] + results.params[1] * X[:,f] #has n entries

  #Calculate feature contributions to overall predictions
  yhat_uncentered = model.predict(X) if predict_proba == False else model.predict_proba(X)[:,1]
  yhat_mean = yhat_uncentered.mean()
  yhat = yhat_uncentered - yhat_mean

  f_int = yhat - f_pdp_interp - f_not_pdp #n entries
  f_pdp_linear = linear_pdp_component #n entries
  f_pdp_nonlinear = f_pdp_interp - f_pdp_linear #n entries

  #Compute MSE's
  whole_pred = f_not_pdp + f_pdp_linear + f_pdp_nonlinear + f_int
  nolinear_pred = f_not_pdp + f_pdp_nonlinear + f_int
  no_nonlinear_pred = f_not_pdp + f_pdp_linear + f_int
  no_int_pred = f_not_pdp + f_pdp_linear + f_pdp_nonlinear
  none_pred = f_not_pdp

  h2 = np.sum((yhat - f_pdp_interp - f_not_pdp)**2) / np.sum(yhat**2)

  #NOTE: This section helps recover some of the covariance lost when the interaction term is omitted.
  #This attenuates the intreaction estimate when predictors are correlated (but no interaction is actually present)
  #whole_pred = sm.OLS(y, sm.add_constant(np.stack([f_not_pdp,f_pdp_linear,f_pdp_nonlinear,f_int],axis=1))).fit().predict(sm.add_constant(np.stack([f_not_pdp,f_pdp_linear,f_pdp_nonlinear,f_int],axis=1)))
  if partial_out == True:
    nolinear_pred = sm.OLS(whole_pred, sm.add_constant(np.stack([f_not_pdp,f_pdp_nonlinear,f_int],axis=1))).fit(
    ).predict(sm.add_constant(np.stack([f_not_pdp,f_pdp_nonlinear,f_int],axis=1)))

    no_nonlinear_pred = sm.OLS(whole_pred, sm.add_constant(np.stack([f_not_pdp,f_pdp_linear,f_int],axis=1))).fit(
    ).predict(sm.add_constant(np.stack([f_not_pdp,f_pdp_linear,f_int],axis=1)))

    no_int_pred = sm.OLS(whole_pred, sm.add_constant(np.stack([f_not_pdp,f_pdp_linear + f_pdp_nonlinear],axis=1))).fit(
    ).predict(sm.add_constant(np.stack([f_not_pdp,f_pdp_linear + f_pdp_nonlinear],axis=1)))

  #Calculate score
  if metric=='rmse':
    metric = rmse
  if metric=='mse':
    metric = mse

  #Add yhat_mean back to 'un mean center' the predictions
  score = metric(y, yhat_uncentered)
  score_no_linear = metric(y, nolinear_pred + yhat_mean)
  score_no_nonlinear = metric(y, no_nonlinear_pred + yhat_mean)
  score_no_int = metric(y, no_int_pred + yhat_mean)
  score_none = metric(y, none_pred + yhat_mean)

  linear_score = score_no_linear - score
  nonlinear_score = score_no_nonlinear - score
  none_score = score_none - score
  #int_score = score_no_int - score - OLD METHOD
  int_score = none_score - linear_score - nonlinear_score

  return linear_score, nonlinear_score, int_score, none_score, values, \
    f_pdp, f_not_pdp, f_pdp_interp, pdp, h2, weights
    #f_pdp is not mean centered

def run_rgid(X, y, model, metric,
             grid_size = 20,
             partial_out = True,
             h = 200, w = 200, barsize = 10,
             feature_limit = None,
             predict_proba = False,
             pdp2_band_width = 0.10, #quantile width of areas for moderated PDPs
             pdp3_band_width = 0.30,
             threeway_int_missing_threshold = 0.25,
             pdp_ips_trim_q = 0.9, #quantile for trimming propensity score weights
             interaction_quantiles = (0.25, 0.75),
             fontsize=12,
             threeway_int_limit = 10,
             propensity_samples = 1000,
             feature_imp_njobs = 1,
             propensity_njobs = 4,
             threeway_int_njobs = 4,
             allow_hover = True):
  
  #Check that all features have nonzero variance
  if np.any(X.var(axis = 0) == 0):
    print(f'Features {np.where(X.var(axis=0) == 0)[0]} have zero variance. Please filter the dataset and trained model to features with non-zero variance')
    return None

  runningtime_FEATURE_IMP = 0
  runningtime_PDP = 0
  runningtime_PD2ICE = 0
  runningtime_PD2RIVER = 0
  runningtime_PD3ICE = 0
  runningtime_PD3RIVER = 0

  ###PD-AVID RUNTIME##
  timestamp_FEATURE_IMP = timeit.default_timer()

  if isinstance(X, pd.core.frame.DataFrame):
    is_df = True
    fnames = X.columns
  else:
    is_df = False
    fnames = None

  if np.array(y).dtype == 'bool':
    y = 1 * y

  if is_df == True:
    for f in range(X.shape[1]):
      if X.iloc[:,f].dtype=='bool':
        X.iloc[:,f] = X.iloc[:,f].astype('int')

    #Remove any constant features
  X = X.loc[:,np.var(X,axis=0) > 0] if is_df == True else X[:,np.var(X,axis=0) > 0]

  #Define number of features as number of columns in X
  k = X.shape[1]
  k_incl = int(k) if feature_limit is None else int(feature_limit)

  if h < barsize*k_incl*1.1:
    h = barsize*k_incl*1.1

  print('Getting ready to loop through dataset features')
    #Misc prep work
  yhat_uncentered = model.predict(X = X) if predict_proba == False else model.predict_proba(X= X)[:,1]
  yhat_mean = yhat_uncentered.mean()
  yhat = yhat_uncentered - yhat_mean

  #Get PDP matrices for main plot (linear, nonlinear, interaction effects)
  graph_dfs = []
  pdp_x_list = []
  pdp_y_list = []
  pdp_w_list = []
  pdp_not_n_list = [] #denoted 'n' because it has n data points
  pdp_y_interp_n_list = [] #denoted 'n' because it has n data points
  pdp_high_list = []
  pdp_low_list = []
  pdp_ice_list = []
  h2_list = []

  pdp_master_array = np.zeros((X.shape[0], k, 2))

  #Get PDP's for each feature
  delayed_funcs = []
  from joblib import Parallel, delayed
  for f in range(k):
    delayed_funcs.append(
        delayed(feature_importance_scores)(X = X, y = y, model = model, f = f, metric = metric,
        grid_size = grid_size, partial_out = partial_out, predict_proba = predict_proba,
        interaction_quantiles = interaction_quantiles) #X_residuals = f_resids,
    )
  if feature_imp_njobs == 1:
    print('Calculating for each feature in order')
  else:
    print('Calculating for features in parallel with '+str(feature_imp_njobs)+ ' CPUs')
  result_tuples = Parallel(n_jobs = feature_imp_njobs)(delayed_funcs)
  print('Computed '+str(len(result_tuples))+ ' PDPs')

  for f in range(k):
    s_linear, s_nonlinear, s_int, s_none, pdp_x, pdp_y, pdp_not_n, pdp_y_interp_n, pdp_ice, h2, weights = result_tuples[f]

    pdp_x_list.append(pdp_x)
    pdp_y_list.append(pdp_y)
    pdp_w_list.append(weights)
    pdp_not_n_list.append(pdp_not_n)
    pdp_y_interp_n_list.append(pdp_y_interp_n)
    pdp_ice_list.append(pdp_ice)
    h2_list.append(h2)

    pdp_master_array[:, f, 0] = pdp_y_interp_n
    pdp_master_array[:, f, 1] = pdp_not_n

    graph_dfs.append( pd.DataFrame({'Feature_Num': str(f), 'Type': ['(a) Linear','(b) Nonlin.','(c) Intxn','None'],
                                      'Diff': [s_linear, s_nonlinear, s_int, s_none]}) )


  #Create PDP df for main graph
  graph_df_full = pd.concat(objs = graph_dfs, axis = 0)
  if fnames is None:
    graph_df_full['Feature'] = ['X'+str(n) for n in graph_df_full['Feature_Num'].astype('int').values]
  else:
    graph_df_full['Feature'] = [fnames[n] for n in graph_df_full['Feature_Num'].astype('int').values]

  #Find top n features from graph_df (will use to limit interaction loop)
  graph_df = graph_df_full[graph_df_full['Type']!='None']
  ranked = graph_df.groupby(['Feature','Feature_Num'],as_index = False)['Diff'].apply(lambda x: np.clip(x, a_min = 0, a_max = None).sum())
  ranked['rank'] = ranked['Diff'].rank(ascending = False)
  if feature_limit is not None:
    kept_features = ranked.loc[ranked['rank']<=feature_limit]['Feature'].values
    kept_feature_nums = ranked.loc[ranked['rank']<=feature_limit]['Feature_Num'].values
    kept_feature_nums = [int(k) for k in kept_feature_nums]
    ranked = ranked[ranked['Feature'].isin(kept_features)]
  else:
    kept_features = X.columns if is_df==True else ['X'+str(i) for i in range(k)]
    kept_feature_nums = np.arange(k)
    #ranked now only has kept features in it

  runningtime_FEATURE_IMP += timeit.default_timer() - timestamp_FEATURE_IMP

  ###PDP RUNTIME###
  timestamp_PDP = timeit.default_timer()

  #Create df for pdp plot
  ns = [len(p) for p in pdp_x_list]
  if fnames is None:
    features = np.concatenate([['X'+str(i)]*n for i,n in zip(range(k), ns)])
  else:
    features = np.concatenate([[fnames[i]]*n for i,n in zip(range(k), ns)])

  feature_num = np.concatenate([[i]*n for i,n in zip(range(k), ns)])
  pdp_df = pd.DataFrame({'Feature': features, 'Feature_Num': feature_num, \
                         'X': np.concatenate(pdp_x_list), 'Y': np.concatenate(pdp_y_list)})

  #Duplicate PDP df for 'linear' and 'nonlinear', since the same PDP will be shown for both hover actions
  pdp_df['Type'] = '(a) Linear' ; pdp_df2 = pdp_df.copy(deep = True) ; pdp_df2['Type'] = '(b) Nonlinear'
  pdp_df_combined = pd.concat(objs = [pdp_df,pdp_df2], axis = 0)

  #Find top n features from pdp plot
  if feature_limit is not None:
    pdp_df_combined = pdp_df_combined[pdp_df_combined['Feature'].isin(kept_features)]

  runningtime_PDP += timeit.default_timer() - timestamp_PDP

  ###PD2ICE and PD3ICE RUNTIME###
  timestamp_PD2ICE_PD3ICE = timeit.default_timer()

  print('Getting propensity weights')
    #Get filters and weights for estimated pairwise h^2 statistics
  h2_filter2_list = []
  h2_weight2_list = []
  h2_filter3_list = []
  h2_weight3_list = []
  delayed_funcs = []
  for f in range(k):
    delayed_funcs.append(
        delayed(propensity_filters_and_weights)(f = f, pdp_w_list = pdp_w_list,
            pdp2_band_width = pdp2_band_width, pdp3_band_width = pdp3_band_width,
            X = X, pdp_ips_trim_q = pdp_ips_trim_q, propensity_samples = propensity_samples) #X_residuals = f_resids,
    )
  if propensity_njobs == 1:
    print('Getting propensity weights in order')
  else:
    print('Getting propensity weights in parallel with '+str(propensity_njobs)+ ' CPUs')
  result_tuples = Parallel(n_jobs = propensity_njobs)(delayed_funcs)
  print('Computed '+str(len(result_tuples))+ ' PDPs')

  for f in range(k):
    f_filters2, f_weights2, f_filters3, f_weights3 = result_tuples[f]
    h2_filter2_list.append(f_filters2); h2_weight2_list.append(f_weights2)
    h2_filter3_list.append(f_filters3); h2_weight3_list.append(f_weights3)

  added_time = timeit.default_timer() - timestamp_PD2ICE_PD3ICE
  runningtime_PD2ICE += added_time / 2
  runningtime_PD3ICE += added_time / 2

  ###PD2RIVER RUNTIME###
  timestamp_PD2RIVER = timeit.default_timer()
    #Get filters and weights for moderated PDPs
  upper_cutoffs = np.quantile(X, axis = 0, q = interaction_quantiles[1])
  lower_cutoffs = np.quantile(X, axis = 0, q = interaction_quantiles[0])
  maxs = np.max(X, axis = 0)
  mins = np.min(X, axis = 0)

  #If they are equal, and at max, then u is unchanged and l adjusts downward
  #If they are equal, and at min, then u is changed and l is unchanged
  upper_cutoffs = np.array([u + 0.001 if (u == l) and (u != m) else u for u, l, m in zip(upper_cutoffs , lower_cutoffs, maxs)])
  lower_cutoffs = np.array([l - 0.001  if (l == u) and (l != m) else l for u, l, m in zip(upper_cutoffs , lower_cutoffs, mins)])

  upper_cutoffs = upper_cutoffs.reshape(1,-1)
  lower_cutoffs = lower_cutoffs.reshape(1,-1)

  binary_adjust_counter = 0
  for f in range(np.array(X).shape[1]):
    if len(np.unique(np.array(X)[:,f]))==2:
      binary_adjust_counter+=1
      upper_cutoffs[0,f] = np.max(X.iloc[:,f]) if is_df == True else np.max(X[:,f])
      lower_cutoffs[0,f] = np.min(X.iloc[:,f]) if is_df == True else np.min(X[:,f])
  print(f'Adjusted upper/lower cutoffs for {binary_adjust_counter} binary features')

  pdp_high_filter = np.array(X) >= upper_cutoffs
  pdp_low_filter = np.array(X) <= lower_cutoffs

  pdp_high_weights = np.zeros(pdp_high_filter.shape)
  pdp_low_weights = np.zeros(pdp_low_filter.shape)

  pdp_high_mat = np.zeros(pdp_high_filter.shape)
  pdp_low_mat = np.zeros(pdp_low_filter.shape)

  print('Getting 2PDQUIVER propensity weights')
  for f in range(k):
    high_weights, low_weights = pdp_propensity_filters_and_weights(f, pdp_high_filter, pdp_low_filter,
                                                                   X, pdp_ips_trim_q, propensity_samples)

    pdp_high_weights[:,f] = high_weights
    pdp_low_weights[:,f] = low_weights

    pdp_high_mat[:,f] = 1 * pdp_high_filter[:,f] * pdp_high_weights[:,f] #change the fth column to be the weight (for included points) or zero (for not included points)
    pdp_low_mat[:,f] = 1 * pdp_low_filter[:,f] * pdp_low_weights[:,f] #change the fth column to be the weight (for included points) or zero (for not included points)

  pdp_high_mat_w = pdp_high_mat / np.sum(pdp_high_mat, axis = 0).reshape(1,-1) #normalize the weight matrix
  pdp_low_mat_w = pdp_low_mat / np.sum(pdp_low_mat, axis = 0).reshape(1,-1) #normalize the weight matrix

  for f in range(k):
    f_pdp_high = pdp_ice_list[f] @ pdp_high_mat_w
    f_pdp_low = pdp_ice_list[f] @ pdp_low_mat_w
    pdp_high_list.append(f_pdp_high)
    pdp_low_list.append(f_pdp_low)

    #Get dataframe of moderated PDPs
  int_score_diffs = []
  f1 = []
  f2 = []
  tuples = []
  if metric=='rmse':
    metric = rmse #callable function
  if metric=='mse':
    metric = mse

  error_matrices = {}
  counter = -1

  if fnames is None:
    f1 = ['X'+str(i) for i in range(k) for j in range(k)]
    f2 = ['X'+str(j) for i in range(k) for j in range(k)]
  else:
    f1 = [fnames[i] for i in range(k) for j in range(k)]
    f2 = [fnames[j] for i in range(k) for j in range(k)]

  X_col = np.concatenate([pdp_x_list[i] for i in range(k) for j in range(k)])
  ns = [len(pdp_x_list[i]) for i in range(k) for j in range(k)]
  int_pdp_col_high = np.concatenate([pdp_high_list[i][:,j] for i in range(k) for j in range(k)])
  int_pdp_col_low = np.concatenate([pdp_low_list[i][:,j] for i in range(k) for j in range(k)])

  pdp_df_high = pd.DataFrame({
      'Feature': np.repeat(f1, ns),
      'Feature 2': np.repeat(f2, ns),
      'Feature Combo': [_f1+':'+_f2 for _f1, _f2 in zip(np.repeat(f1, ns), np.repeat(f2, ns))],
      'X': X_col,
      'Feature 2 Level': ['High']*len(X_col),
      'Y': int_pdp_col_high
  })
  pdp_df_low = pd.DataFrame({
      'Feature': np.repeat(f1, ns),
      'Feature 2': np.repeat(f2, ns),
      'Feature Combo': [_f1+':'+_f2 for _f1, _f2 in zip(np.repeat(f1, ns), np.repeat(f2, ns))],
      'X': X_col,
      'Feature 2 Level': ['Low']*len(X_col),
      'Y': int_pdp_col_low
  })

  pdp_df_twoway = pd.concat([pdp_df_high,pdp_df_low], axis = 0, ignore_index = True)
  pdp_df_twoway = pdp_df_twoway[pdp_df_twoway['Feature'] != pdp_df_twoway['Feature 2']]

  #Limit pdp df to kept features
  if feature_limit is not None:
    pdp_df_twoway = pdp_df_twoway.loc[(pdp_df_twoway['Feature'].isin(kept_features)) & (pdp_df_twoway['Feature 2'].isin(kept_features)),:]
  print('PDP df shape: ' + str(pdp_df_twoway.shape))

  runningtime_PD2RIVER += timeit.default_timer() - timestamp_PD2RIVER

  ###PD2ICE RUNTIME###
  timestamp_PD2ICE = timeit.default_timer()

  print('Running through interactions')
  f1_short = [] ; f2_short = [] #these will only store the interactions we loop through (k*(k-1)/2), not all k**2
  if feature_limit is None:
    kept_feature_nums = np.arange(k)
  feature_int_error_cors = {}
  for i in kept_feature_nums:
    for j in range(i):
      if j in kept_feature_nums:
        counter+=1 #starts at counter = 0
        if counter%500==0:
          print('Interaction '+str(counter))
        if fnames is None:
          f1_short.append('X'+str(i))
          f2_short.append('X'+str(j))
        else:
          f1_short.append(fnames[i])
          f2_short.append(fnames[j])

        name_tuple = np.sort([f1_short[-1], f2_short[-1]])
        f_tuple = (i,j) ; tuples.append(name_tuple[0]+":"+name_tuple[1])
        f1_ice = pdp_ice_list[i]
        f2_ice = pdp_ice_list[j]

        ###################################
        f1_filters, f1_weights = h2_filter2_list[i], h2_weight2_list[i]
        f2_filters, f2_weights = h2_filter2_list[j], h2_weight2_list[j]

        pdp2_f1_f2 = np.array([np.average(f1_ice[:,f], axis = 1, weights = w[f]) for f,w in zip(f2_filters, f2_weights)]).T #f1 increases down, f2 increases across
        pdp2_f2_f1 = np.array([np.average(f2_ice[:,f], axis = 1, weights = w[f]) for f,w in zip(f1_filters, f1_weights)]) #f2 increases across, f1 increases down
        pdp2_final = (pdp2_f1_f2 + pdp2_f2_f1) / 2

          #Calculate H^2 score
        pdp2_final = pdp2_final - pdp2_final.mean()
        pdp2_f1 = (pdp2_final.mean(axis = 1) - pdp2_final.mean(axis = 1).mean()).reshape(-1,1)
        pdp2_f2 = (pdp2_final.mean(axis = 0) - pdp2_final.mean(axis = 0).mean()).reshape(1,-1)

        resid_h2 = np.sum((pdp2_final - pdp2_f1 - pdp2_f2)**2) / np.sum(pdp2_final**2)

        #################

        #Interaction-length list, where each element is a feature-length list
        int_score_diffs.append(resid_h2)


  int_df = pd.DataFrame({'Feature': f1_short, 'Feature 2': f2_short, 'Feature Combo': tuples, 'Diff': int_score_diffs})
  #At this point, int_df is missing the 'reversed' copies of itself. That is easy to fix - just flip Feature and Feature 2.
  int_df_copied = int_df.copy(deep = True)
  featcol = int_df['Feature'].values ; feat2col = int_df['Feature 2'].values
  int_df_copied.loc[:,'Feature'] = feat2col ; int_df_copied.loc[:,'Feature 2'] = featcol
  int_df = pd.concat([int_df, int_df_copied], axis = 0)
  int_df.loc[int_df['Feature']==int_df['Feature 2'], 'Diff'] = 0 #zero out same-variable rows; variable can't interact with itself
  orig_int_df = int_df.copy(deep = True)

  runningtime_PD2ICE += timeit.default_timer() - timestamp_PD2ICE

  ###PD3ICE RUNTIME###
  timestamp_PD3ICE = timeit.default_timer()

  #Higher-order interaction plot
  int_df_filt = int_df[int_df['Feature'] != int_df['Feature 2']]
  int_df_filt = int_df_filt[int_df_filt['Diff'] > 0 ]
  int_df_filt = int_df_filt.drop_duplicates('Feature Combo')
  int_df_filt = int_df_filt.sort_values('Diff', ascending = False)
  best_rmse = []
  best_rmse_unadj = []
  best_higherorder_interaction = []
  best_pdp_low = []
  best_pdp_high = []

  if threeway_int_njobs == 1:
    print('Getting higher order interactions weights in order')
  else:
    print('Getting higher order interactions in parallel with '+str(threeway_int_njobs)+ ' CPUs')

  threeway_int_score_matrix = np.zeros((min([int_df_filt.shape[0], threeway_int_limit]), len(kept_feature_nums)))
  avg_cor_matrix = np.zeros((min([int_df_filt.shape[0], threeway_int_limit]), len(kept_feature_nums)))

  int_tuple_name_list = []
  int_tuple_list = []

  delayed_funcs = []
  for ix in range(min([int_df_filt.shape[0], threeway_int_limit])):
    int_tuple_name = ( int_df_filt.iloc[ix,0], int_df_filt.iloc[ix,1] )
    if fnames is None:
      int_tuple = ( int(int_tuple_name[0].replace('X','')), int(int_tuple_name[1].replace('X','')) )
    else:
      int_tuple = (
          int(np.where(np.array(fnames)==int_tuple_name[0])[0]),
          int(np.where(np.array(fnames)==int_tuple_name[1])[0]),
      )
    int_tuple_name_list.append(int_tuple_name)
    int_tuple_list.append(int_tuple)
    f2, f3 = int_tuple[0], int_tuple[1]  #renaming indices as i/j/k to be consistent with pairwise H^2 score
    for f, i in zip(kept_feature_nums, range(len(kept_feature_nums))):
      f_name = 'X'+str(f) if fnames is None else fnames[f]
      if f_name not in int_tuple_name:

        delayed_funcs.append(
        delayed(threeway_h2)(f1_ice = pdp_ice_list[f],
                            f2_ice = pdp_ice_list[f2],
                            f3_ice = pdp_ice_list[f3],
                            f1_filters = h2_filter3_list[f],
                            f2_filters = h2_filter3_list[f2],
                            f3_filters = h2_filter3_list[f3],
                            f1_weights = h2_filter3_list[f],
                            f2_weights = h2_filter3_list[f2],
                            f3_weights = h2_filter3_list[f3],
                            grid_size = grid_size,
                            idf = int_df, f1 = f, f2 = f2, f3 = f3, fnames = fnames,
                            missing_threshold = threeway_int_missing_threshold)
    )
  threeway_h2_results = Parallel(n_jobs = threeway_int_njobs)(delayed_funcs)
  #Now append results to threeway_int_score_matrix
  counter = -1
  for ix in range(min([int_df_filt.shape[0], threeway_int_limit])):
    int_tuple_name = ( int_df_filt.iloc[ix,0], int_df_filt.iloc[ix,1] )
    # if fnames is None:
    #   int_tuple = ( int(int_tuple_name[0].replace('X','')), int(int_tuple_name[1].replace('X','')) )
    # else:
    #   int_tuple = (np.where(np.array(fnames)==int_tuple_name[0])[0], np.where(np.array(fnames)==int_tuple_name[1])[0])
    # f2, f3 = int_tuple[0], int_tuple[1]
    for f, i in zip(kept_feature_nums, range(len(kept_feature_nums))):
      f_name = 'X'+str(f) if fnames is None else fnames[f]
      if f_name not in int_tuple_name:
        counter += 1
        threeway_int_score_matrix[ix, i] = threeway_h2_results[counter][0]
        avg_cor_matrix[ix,i] = threeway_h2_results[counter][1]


  higher_order_df = pd.DataFrame({'Feature': list(kept_features) * threeway_int_score_matrix.shape[0],
                                     'Feature Num': list(kept_feature_nums) * threeway_int_score_matrix.shape[0],
                                     'Score': threeway_int_score_matrix.reshape(-1),
                                     'Interaction': np.repeat([str(_int) for _int in int_tuple_name_list], len(kept_features)),
                                     'Interaction V1': np.repeat([_int[0] for _int in int_tuple_list], len(kept_features)),
                                     'Interaction V2': np.repeat([_int[1] for _int in int_tuple_list], len(kept_features)),
                                     'Feature Combo': list(zip(
                                        list(kept_feature_nums) * threeway_int_score_matrix.shape[0],
                                        np.repeat([_int[0] for _int in int_tuple_list], len(kept_features)),
                                        np.repeat([_int[1] for _int in int_tuple_list], len(kept_features))
                                     ))})
    #Supplemental code to fully represent biggest interactions
  topq = 0.05
  top_df = higher_order_df.sort_values('Score', ascending = False).iloc[0:int(higher_order_df.shape[0]*topq),:]
  top_threeway_ints = top_df['Feature Combo']

  #Swap features 0 and 1
    #Re-do code below when using column names
  top_df2 = top_df.copy(deep = True)
  top_df2['Feature Num'] = [t[1] for t in top_threeway_ints] #put second feature as main feature
  top_df2['Feature'] = ['X'+ str(n) for n in list(top_df2['Feature Num'])] if fnames is None else [fnames[n] for n in list(top_df2['Feature Num'])]
  top_df2['Interaction V1'] = top_df['Feature Num'] #put first feature as second feature
  if fnames is None:
    top_df2['Interaction'] = [str(('X'+str(i1), 'X'+str(i2))) for i1, i2 in zip(top_df2['Interaction V1'], top_df2['Interaction V2'])]
  else:
    top_df2['Interaction'] = [str((fnames[i1], fnames[i2])) for i1, i2 in zip(top_df2['Interaction V1'], top_df2['Interaction V2'])]

  #Swap features 0 and 2
     #Re-do code below when using column names
  top_df3 = top_df.copy(deep = True)
  top_df3['Feature Num'] = [t[2] for t in top_threeway_ints] #put third feature as main feature
  top_df3['Feature'] = ['X'+ str(n) for n in list(top_df3['Feature Num'])] if fnames is None else [fnames[n] for n in list(top_df3['Feature Num'])]
  top_df3['Interaction V2'] = top_df['Feature Num'] #put first feature as third feature
  if fnames is None:
    top_df3['Interaction'] = [str(('X'+str(i1), 'X'+str(i2))) for i1, i2 in zip(top_df3['Interaction V1'], top_df3['Interaction V2'])]
  else:
    top_df3['Interaction'] = [str((fnames[i1], fnames[i2])) for i1, i2 in zip(top_df3['Interaction V1'], top_df3['Interaction V2'])]

  higher_order_df = pd.concat([higher_order_df, top_df2, top_df3]).drop_duplicates(['Feature','Interaction V1','Interaction V2'])

  runningtime_PD3ICE += timeit.default_timer() - timestamp_PD3ICE

  ###PD3RIVER RUNTIME###
  timestamp_PD3RIVER = timeit.default_timer()

  #Create matrix of moderated PDPs
  print('Filtering three-way interactions for PDP matrix')
  high_int_dfs = []
  higher_order_df_filtered_for_pdps = higher_order_df[higher_order_df['Score'] > 0]
  max_int_n = int(np.floor( 5000 / (grid_size*4) ))
  if higher_order_df_filtered_for_pdps.shape[0] > max_int_n:
    higher_order_df_filtered_for_pdps = higher_order_df_filtered_for_pdps.sort_values('Score', ascending = False).iloc[0:max_int_n,:]

  print('Calculated three-way moderated PDPs')
  for i in range(higher_order_df_filtered_for_pdps.shape[0]):
    f_name = higher_order_df_filtered_for_pdps['Feature'].iloc[i]
    f = higher_order_df_filtered_for_pdps["Feature Num"].iloc[i]
    i1 = higher_order_df_filtered_for_pdps["Interaction V1"].iloc[i]
    i2 = higher_order_df_filtered_for_pdps["Interaction V2"].iloc[i]

    int_filter_high_high = pdp_high_filter[:,i1] & pdp_high_filter[:,i2]
    int_filter_low_high = pdp_low_filter[:,i1] & pdp_high_filter[:,i2]
    int_filter_high_low = pdp_high_filter[:,i1] & pdp_low_filter[:,i2]
    int_filter_low_low = pdp_low_filter[:,i1] & pdp_low_filter[:,i2]

    #Only compute two-way PDP if all four high/low combinations are available
    ice_matrix = pdp_ice_list[f]
    x = pdp_df_combined[pdp_df_combined['Feature']==f_name]['X'].unique()

    high_int_df_combined = []
    if (int_filter_high_high.sum() > 0):
      int_weights_high_high = (pdp_high_weights[:, i1] * pdp_high_weights[:,i2])[int_filter_high_high]
      int_pdp_high_high = np.average(ice_matrix[:,int_filter_high_high], axis = 1, weights = int_weights_high_high)
      #int_pdp_high_high = int_pdp_high_high - int_pdp_high_high.mean()
      high_int_df_HIGH_HIGH = pd.DataFrame({'Feature': f_name, 'V2 Level': 'High', 'V3 Level': 'High', 'X': x, 'Y': int_pdp_high_high, 'Interaction V1': i1, 'Interaction V2': i2})
      high_int_df_combined.append(high_int_df_HIGH_HIGH)

    if (int_filter_low_high.sum() > 0):
      int_weights_low_high = (pdp_low_weights[:, i1] * pdp_high_weights[:, i2])[int_filter_low_high]
      int_pdp_low_high = np.average(ice_matrix[:,int_filter_low_high], axis = 1, weights = int_weights_low_high)
      #int_pdp_low_high = int_pdp_low_high - int_pdp_low_high.mean()
      high_int_df_LOW_HIGH = pd.DataFrame({'Feature': f_name, 'V2 Level': 'Low', 'V3 Level': 'High', 'X': x, 'Y': int_pdp_low_high, 'Interaction V1': i1, 'Interaction V2': i2})
      high_int_df_combined.append(high_int_df_LOW_HIGH)


    if (int_filter_high_low.sum() > 0):
      int_weights_high_low = (pdp_high_weights[:, i1] * pdp_low_weights[:, i2])[int_filter_high_low]
      int_pdp_high_low = np.average(ice_matrix[:,int_filter_high_low], axis = 1, weights = int_weights_high_low)
      #int_pdp_high_low = int_pdp_high_low - int_pdp_high_low.mean()
      high_int_df_HIGH_LOW = pd.DataFrame({'Feature': f_name, 'V2 Level': 'High', 'V3 Level': 'Low', 'X': x, 'Y': int_pdp_high_low, 'Interaction V1': i1, 'Interaction V2': i2})
      high_int_df_combined.append(high_int_df_HIGH_LOW)

    if (int_filter_low_low.sum() > 0):
      int_weights_low_low = (pdp_low_weights[:, i1] * pdp_low_weights[:, i2])[int_filter_low_low]
      int_pdp_low_low = np.average(ice_matrix[:,int_filter_low_low], axis = 1, weights = int_weights_low_low)
      #int_pdp_low_low = int_pdp_low_low - int_pdp_low_low.mean()
      high_int_df_LOW_LOW = pd.DataFrame({'Feature': f_name, 'V2 Level': 'Low', 'V3 Level': 'Low', 'X': x, 'Y': int_pdp_low_low, 'Interaction V1': i1, 'Interaction V2': i2})
      high_int_df_combined.append(high_int_df_LOW_LOW)

    if len(high_int_df_combined)>0:
      high_int_dfs.append(pd.concat(high_int_df_combined, axis = 0))

  if len(high_int_dfs) > 0:
    high_int_df_long = pd.concat(high_int_dfs, axis = 0)
    print(f'Three-way PDP df has {high_int_df_long.shape[0]} rows')
  else:
    print('No three-way PDPs to visualize')
    high_int_df_long = pd.DataFrame(columns = ['Feature','V2 Level','V3 Level','X','Y','Interaction V1','Interaction V2'])

  runningtime_PD3RIVER += timeit.default_timer() - timestamp_PD3RIVER

  ####PLOT RESULTS###

  ###PD-AVID RUNTIME - PLOTTING###
  timestamp_FEATURE_IMP = timeit.default_timer()

  #Plot results
  print('Generating plot')
  feature_selector1 = alt.selection_single(on="mouseover", encodings=['y'])
  feature_selector1_click = alt.selection_multi(on="click", encodings=['y'])
  feature_selector2 = alt.selection_single(on="mouseover", encodings = ['color'])
  feature_selector2_click = alt.selection_multi(on="click", encodings = ['color'])
  feature_selector3 = alt.selection_single(on="mouseover", encodings=['y'])
  feature_selector3_click = alt.selection_multi(on="click", encodings=['y'])
  feature_selector4 = alt.selection_single(on="mouseover", encodings=['x','y'])
  feature_selector4_click = alt.selection_multi(on="click", encodings=['x','y'])

  #global _plot_feature_selectors
  _plot_feature_selectors = [feature_selector1, feature_selector1_click, feature_selector2,
                              feature_selector2_click, feature_selector3, feature_selector3_click, feature_selector4, feature_selector4_click]

  dom = ['(a) Linear','(b) Nonlin.','(c) Intxn']
  ran = ['#1f77b4FF', '#17becfFF', '#2ca02cFF']

  #Bar charts
  sort_order = ranked.sort_values('rank', ascending = True)['Feature'].values
    #limited to top n if feature_limit is not none
  if feature_limit is not None:
    graph_df = graph_df[graph_df['Feature'].isin(kept_features)]

    #This is the left-most feature importance decomposition chart
  decomp_xmin = graph_df.groupby('Feature')['Diff'].apply(lambda x: x[x<0].sum()).min()
  decomp_xmax = graph_df.groupby('Feature')['Diff'].apply(lambda x: x[x>0].sum()).max()

  _opac = alt.condition(feature_selector1|feature_selector1_click, alt.value(1.0), alt.value(0.5)) if allow_hover == True else alt.condition(feature_selector1_click, alt.value(1.0), alt.value(0.5))
  decomp_chart = alt.Chart(graph_df, title = 'Plot 1: Feature Importances').mark_bar(size=barsize).encode(
      x=alt.X('sum(Diff)', scale = alt.Scale(domain = (decomp_xmin, decomp_xmax)), axis = alt.Axis(title = 'Score')),
      y=alt.Y('Feature',sort = sort_order),
      color=alt.Color('Type', scale = alt.Scale(domain = dom, range = ran),
                      legend=alt.Legend(title = None, orient='top', labelFontSize = fontsize, titleFontSize = fontsize)),
      opacity = _opac
      ).properties(
      width=w,
      height=h
  )
  decomp_chart = decomp_chart.add_selection(feature_selector1, feature_selector1_click) if allow_hover == True else decomp_chart.add_selection(feature_selector1_click)

  runningtime_FEATURE_IMP += timeit.default_timer() - timestamp_FEATURE_IMP
  timestamp_PDP = timeit.default_timer()

  ###PDP RUNTIME - PLOTTING###
  pdp_xmin = pdp_df_combined['Y'].min()
  pdp_xmax = pdp_df_combined['Y'].max()

  pdp_chart = alt.Chart(pdp_df_combined, title = 'Plot 2: Partial Dep. Plots (PDPs)').mark_line().encode(
      alt.X('X'),
      alt.Y('Y', scale = alt.Scale(zero = False, domain = (pdp_xmin, pdp_xmax))),
      color=alt.condition(feature_selector2|feature_selector2_click,alt.Color('Feature:N',legend = None),alt.value('lightgray'))
      ).properties(width=w,height=h)

  pdp_chart = pdp_chart.transform_filter(feature_selector1 | feature_selector1_click) if allow_hover == True else pdp_chart.transform_filter(feature_selector1_click)

  pdp_chart = pdp_chart.add_selection(
              feature_selector2,feature_selector2_click)

  runningtime_PDP += timeit.default_timer() - timestamp_PDP

  ###PD2ICE RUNTIME - PLOTTING###
  timestamp_PD2ICE = timeit.default_timer()

  scale = alt.Scale(domain = int_df['Feature'].unique(),
                                    range = list([int_color]*k))
  int_df_positive = int_df[int_df['Diff'] > 0]
  if int_df_positive.shape[0] > 5000:
    int_df_positive = int_df_positive.sort_values('Diff', ascending = False).iloc[0:5000,:]
    print('Filtering to '+str(int_df_positive.shape[0])+ ' interaction scores')

  _col = alt.condition(feature_selector3|feature_selector3_click,
                          alt.Color('Feature:N',scale = scale,legend = None),
                          alt.value('lightgray')) if allow_hover == True else alt.condition(feature_selector3_click,
                          alt.Color('Feature:N',scale = scale,legend = None),
                          alt.value('lightgray'))
  int_chart = alt.Chart(data = int_df_positive, title = 'Plot 3: Pairwise Intxn Scores').mark_bar(size = barsize).encode(
      x=alt.X('sum(Diff)',
      axis = alt.Axis(title = 'Score')),
      y=alt.Y('Feature 2', title = 'Interaction Feature', sort = sort_order),
      color=_col
      ).properties(width=w,height=h)

  int_chart = int_chart.transform_filter(feature_selector1 | feature_selector1_click) if allow_hover == True else int_chart.transform_filter(feature_selector1_click)
  int_chart = int_chart.add_selection(feature_selector3, feature_selector3_click)

  runningtime_PD2ICE += timeit.default_timer() - timestamp_PD2ICE

  ###PD2RIVER RUNTIME - PLOTTING###
  timestamp_PD2RIVER = timeit.default_timer()
  int_names = list(error_matrices.keys())
  positive_ints = list(int_df[int_df['Diff']>0]['Feature Combo'])
  positive_ints = positive_ints + [p.split(':')[1]+':'+p.split(':')[0] for p in positive_ints]

  pdp_df_twoway_incl = pdp_df_twoway[pdp_df_twoway['Feature Combo'].isin(positive_ints)]
  print('Filtered PDP df has '+str(pdp_df_twoway_incl.shape[0])+ ' rows')
  print('based on '+ str(pdp_df_twoway_incl['Feature Combo'].nunique())+ ' interactions with positive score')

  max_int_n = int(np.floor( 5000 / (grid_size*2*2) )) #each interaction requires 2*grid_size rows, and each interaction can appear twice
  if pdp_df_twoway_incl.shape[0] > 5000:
      #Get top interactions (note: each interaction is counted twice, because it can be visualized two ways)
    top_n_ints = list(int_df.sort_values('Diff', ascending = False)['Feature Combo'].iloc[0:max_int_n])
      #This includes both 'copies' of the interaction, since pdp_df Feature Combos are ordered
    top_n_ints = top_n_ints + [p.split(':')[1]+ ':' + p.split(':')[0] for p in top_n_ints]
    pdp_df_twoway_incl = pdp_df_twoway_incl[pdp_df_twoway_incl['Feature Combo'].isin(top_n_ints)]

    done = False
    counter = -1
    while done == False:
      counter +=1
      next_index = max_int_n + counter
      next_int = int_df.sort_values('Diff', ascending = False)['Feature Combo'].iloc[next_index]
      next_int = [next_int] + [next_int.split(':')[1]+ ':' + next_int.split(':')[0]]
      pdp_df_added = pdp_df_twoway[pdp_df_twoway['Feature Combo'].isin(next_int)]

      if pdp_df_added.shape[0] + pdp_df_twoway_incl.shape[0] <=5000:
        pdp_df_twoway_incl = pd.concat([pdp_df_twoway_incl, pdp_df_added], axis = 0)
      else:
        done = True

    print('Filtered to '+str(pdp_df_twoway_incl['Feature Combo'].nunique()) + ' interactions')
    print('PDP df now has '+str(pdp_df_twoway_incl.shape[0]) + ' rows')

  ymin = pdp_df_twoway_incl['Y'].min()
  ymax = pdp_df_twoway_incl['Y'].max()

  color_dom = list(pdp_df_twoway_incl['Feature'].unique())
  color_ran = ['#1f77b4BF'] * len(color_dom)

  pdp_twoway_plot = alt.Chart(pdp_df_twoway_incl, title = 'Plot 4: Pairwise PDPs (-- V2 Low, — V2 High)').mark_line().encode(
  x = alt.X('X'),
  y = alt.Y('Y', title = 'Y', scale=alt.Scale(domain=(ymin,ymax))),
  color = alt.Color('Feature:N', scale = alt.Scale(domain = color_dom, range = color_ran), legend = None),
  detail = alt.Detail('Feature 2'),
  strokeDash = alt.StrokeDash('Feature 2 Level',
    legend = None # alt.Legend(orient= 'top', title = 'Feature 2 Value', labelFontSize = fontsize, titleFontSize = fontsize)
    )
  ).properties(
    width=w,
    height=h
  )
  if allow_hover == True:
    pdp_twoway_plot = pdp_twoway_plot.transform_filter(
      feature_selector1 | feature_selector1_click).transform_filter(
          feature_selector3 | feature_selector3_click)
  else:
    pdp_twoway_plot = pdp_twoway_plot.transform_filter(feature_selector1_click).transform_filter(feature_selector3_click)

  runningtime_PD2RIVER += timeit.default_timer() - timestamp_PD2RIVER


  ###PD3ICE RUNTIME - PLOTTING###
  timestamp_PD3ICE = timeit.default_timer()

  h2_min, h2_max = higher_order_df.Score.min(), higher_order_df.Score.quantile(0.99)
  #Get feature names
  higher_order_df['Variable 2'] = ['X'+str(i) for i in list(higher_order_df['Interaction V1'])] if fnames is None else [fnames[i] for i in list(higher_order_df['Interaction V1'])]
  higher_order_df['Variable 3'] = ['X'+str(i) for i in list(higher_order_df['Interaction V2'])] if fnames is None else [fnames[i] for i in list(higher_order_df['Interaction V2'])]
  higher_order_df = higher_order_df[ (higher_order_df['Feature Num'] != higher_order_df['Interaction V1'] ) & (higher_order_df['Feature Num'] != higher_order_df['Interaction V2'])]
  sort_order_v2 = ['X'+str(i) for i in np.sort(higher_order_df['Interaction V1'].unique())] if fnames is None else [fnames[i] for i in np.sort(higher_order_df['Interaction V1'].unique())]
  sort_order_v3 = ['X'+str(i) for i in np.sort(higher_order_df['Interaction V2'].unique())] if fnames is None else [fnames[i] for i in np.sort(higher_order_df['Interaction V2'].unique())]

  _opac3 = alt.condition(feature_selector4|feature_selector4_click, alt.value(1.0), alt.value(0.5)) if allow_hover == True else alt.condition(feature_selector4_click, alt.value(1.0), alt.value(0.5))
  high_int_chart = alt.Chart(higher_order_df[['Score','Feature','Interaction','Variable 2','Variable 3']],
                             title = 'Plot 5: Three-way Intxn Scores').mark_rect().encode(
      y = alt.Y('Variable 2', sort = sort_order_v2), x = alt.X('Variable 3', sort = sort_order_v3),
      color = alt.Color('sum(Score):Q',
                    legend = alt.Legend(orient = 'bottom', title = 'Interaction Score', labelFontSize = fontsize),
                        scale = alt.Scale(domain = (h2_min, h2_max))
                    ),
      opacity = _opac3,
      detail = alt.Detail('Feature:N'),
      tooltip = 'sum(Score):Q').properties(width = w, height = h) #Interaction:N
  high_int_chart = high_int_chart.add_selection(feature_selector4, feature_selector4_click)
  high_int_chart = high_int_chart.transform_filter(feature_selector1 | feature_selector1_click) if allow_hover == True else high_int_chart.transform_filter(feature_selector1_click)

  runningtime_PD3ICE += timeit.default_timer() - timestamp_PD3ICE

  ###PD3RIVER RUNTIME - PLOTTING###
  timestamp_PD3RIVER = timeit.default_timer()

  #Create chart showing stratified PDPs by the best two-way interaction for each feature
  ymin = high_int_df_long['Y'].min()
  ymax = high_int_df_long['Y'].max()

  #Get feature names
  if fnames is None:
    high_int_df_long['Variable 2'] = ['X'+str(i) for i in list(high_int_df_long['Interaction V1'])]
    high_int_df_long['Variable 3'] = ['X'+str(i) for i in list(high_int_df_long['Interaction V2'])]
  else:
    high_int_df_long['Variable 2'] = [fnames[i] for i in list(high_int_df_long['Interaction V1'])]
    high_int_df_long['Variable 3'] = [fnames[i] for i in list(high_int_df_long['Interaction V2'])]


  high_int_pdp_chart = alt.Chart(high_int_df_long, title = 'Plot 6: Three-Way PDPs (-- V2 Low, — V2 High)').mark_line().encode(
  x = alt.X('X'),
  y = alt.Y('Y', title = 'Y (Centered)', scale = alt.Scale(domain = (ymin, ymax))),
  #color = alt.Color('Feature:N', scale = alt.Scale(domain = color_dom, range = color_ran), legend = None),
  detail = 'Feature:N',
  strokeDash = alt.StrokeDash('V2 Level', legend = None),
  color = alt.Color('V3 Level', legend = alt.Legend(orient='right', title = 'V3 Level'))
  ).properties(width = w, height = h)
  high_int_pdp_chart = high_int_pdp_chart.transform_filter(feature_selector4 | feature_selector4_click) if allow_hover == True else high_int_pdp_chart.transform_filter(feature_selector4_click)
  high_int_pdp_chart = high_int_pdp_chart.transform_filter(feature_selector1 | feature_selector1_click) if allow_hover == True else high_int_pdp_chart.transform_filter(feature_selector1_click)

  runningtime_PD3RIVER += timeit.default_timer() - timestamp_PD3RIVER
  timestamp_FEATURE_IMP = timeit.default_timer()

  final_plot =alt.vconcat(
    alt.hconcat(decomp_chart, pdp_chart).resolve_scale(color='independent'),
    alt.hconcat(int_chart, pdp_twoway_plot).resolve_scale(color='independent'),
    alt.hconcat(high_int_chart, high_int_pdp_chart).resolve_scale(color='independent')
  ).resolve_scale(color='independent'
  ).configure_axis(labelFontSize=fontsize,titleFontSize=fontsize
                  ).configure_title(fontSize=fontsize).configure_view(stroke=None)

  runningtime_FEATURE_IMP += timeit.default_timer() - timestamp_FEATURE_IMP
  print('Feature imp runtime: '+str(runningtime_FEATURE_IMP))
  print('PDP runtime: '+str(runningtime_PDP))
  print('PD2ICE runtime: '+str(runningtime_PD2ICE))
  print('PD2RIVER runtime: '+str(runningtime_PD2RIVER))
  print('PD3ICE runtime: '+str(runningtime_PD3ICE))
  print('PD3RIVER runtime: '+str(runningtime_PD3RIVER))
  runtime_dict = {'Feature imp runtime': runningtime_FEATURE_IMP,
                  'PDP runtime': runningtime_PDP,
                  'PD2ICE runtime': runningtime_PD2ICE,
                  'PD2RIVER runtime': runningtime_PD2RIVER,
                  'PD3ICE runtime': runningtime_PD3ICE,
                  'PD3RIVER runtime': runningtime_PD3RIVER}

  return {'plot': final_plot}

    #Can uncomment the return statement below for debugging purposes
  # return {'plot': final_plot, 'interactivity': _plot_feature_selectors, 'pdp_df_oneway': pdp_df_combined,
  #         'pdp_df_twoway': pdp_df_twoway, 'int_df': int_df,
  #         'orig_int_df': orig_int_df,
  #         'feature_ranks': ranked,
  #         'graph_df': graph_df_full,'pdp_ice_list': pdp_ice_list,
  #         'runtime_dict': runtime_dict,
  #         'threeway_int_tuples': int_tuple_list,
  #         'threeway_int_score_matrix': threeway_int_score_matrix,
  #         'pdp_df_threeway': high_int_df_long,
  #         'higher_order_df': higher_order_df,
  #         'H2': h2_list,
  #         'h2_filter2_list': h2_filter2_list,
  #         'h2_weight2_list': h2_weight2_list,
  #         'h2_filter3_list': h2_filter3_list,
  #         'h2_weight3_list': h2_weight3_list,
  #         'avg_cor_matrix': avg_cor_matrix
  #         }
