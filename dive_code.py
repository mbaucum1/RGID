def cutpoints(var, n_points): #cutpoints are MIDPOINTS of evenly divided segments
  width = 1 / n_points
  quantiles = np.arange(0, 1, width) + width/2
  weights = np.ones(n_points) / n_points
  values = np.array([np.quantile(var, q, method = 'inverted_cdf') for q in quantiles])
  values = np.round(values, 3) #rounding allows us to lump similar quantiles together
  df = pd.DataFrame({'values': values, 'weights': weights})
  return df.groupby('values', as_index = False)['weights'].sum()

def rmse(y_true, y_pred):
  return np.sqrt(np.mean((y_true - y_pred)**2))
def mse(y_true, y_pred):
  return np.mean((y_true - y_pred)**2)


def feature_importance_scores_v1(X, y, model, f, metric, higher_is_better = False,
                                 n_points = 20, predict_proba = False):

  #Choose a continuous feature, then make predictions for all imputed values of that feature
  if isinstance(X, pd.core.frame.DataFrame):
    is_df = True
    fnames = X.columns
  else:
    is_df = False

  c = cutpoints(X.iloc[:,f], n_points = n_points) if is_df == True else cutpoints(X[:,f], n_points = n_points)
  weights = c['weights'].values
  values = c['values'].values

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
  
  del X_impute
  
    #Create one-way PDP's by averaging across both axes
  f_not_pdp = np.average(pdp, axis = 0, weights = weights) 
  f_pdp = pdp.mean(axis=1).reshape(-1)
  f_pdp_interp = np.interp(X.iloc[:,f], values, f_pdp) if is_df == True else np.interp(X[:,f], values, f_pdp)

    #Center both pdps
  f_not_pdp = f_not_pdp - f_not_pdp.mean()
  f_pdp_interp = f_pdp_interp - f_pdp_interp.mean()

  #Calculate feature contributions to overall predictions
  yhat_uncentered = model.predict(X) if predict_proba == False else model.predict_proba(X)[:,1]
  yhat_mean = yhat_uncentered.mean()
  yhat = yhat_uncentered - yhat_mean

  #Calculate score differences
  if metric=='rmse':
    metric = rmse
  if metric=='mse':
    metric = mse

  #Compute feature importance scores (note that these scores use the un-centered PDPs)
  if higher_is_better == False:
    feature_score = metric(y, f_not_pdp + f_not_pdp.mean()) - metric(y, yhat_uncentered)
  else:
    feature_score = metric(y, yhat_uncentered) - metric(y, f_not_pdp + f_not_pdp.mean())

  return feature_score, values, f_pdp, f_not_pdp, f_pdp_interp

#Dashboard plot posted to Github after INFORMS (feature importance and PDP only)
#Run feature importance function
def dive_dashboard_v1(X, y, model, metric, higher_is_better = False, predict_proba = False, 
                            pdp_n_points = 20, h = 200, w = 200, barsize = 10,
                            fontsize=12):

  if isinstance(X, pd.core.frame.DataFrame):
    is_df = True
    fnames = X.columns
  else:
    is_df = False
    fnames = None

  #Define number of features as number of columns in X
  k = X.shape[1]
  if h < barsize*k:
    h = barsize*k*1.1

  #Get PDP matrices for main plot (linear, nonlinear, interaction effects)
  graph_dfs = []
  pdp_x_list = []
  pdp_y_list = []
  pdp_not_n_list = [] #denoted 'n' because it has n data points
  pdp_y_interp_n_list = [] #denoted 'n' because it has n data points

  pdp_master_array = np.zeros((X.shape[0], k, 2))

  #Get PDP's for each feature
  for f in range(k):
    print('Calculating for feature '+str(f+1))

    score, pdp_x, pdp_y, pdp_not_n, pdp_y_interp_n = feature_importance_scores_v1(
        X = X, y = y, model = model, f = f, metric = metric, higher_is_better = higher_is_better,
        n_points = pdp_n_points, predict_proba = predict_proba) #f_resids
    pdp_x_list.append(pdp_x)
    pdp_y_list.append(pdp_y)
    pdp_not_n_list.append(pdp_not_n)
    pdp_y_interp_n_list.append(pdp_y_interp_n)
    
    pdp_master_array[:, f, 0] = pdp_y_interp_n
    pdp_master_array[:, f, 1] = pdp_not_n
    
    graph_dfs.append( pd.DataFrame({'Feature_Num': [str(f)], 'Diff': [score]}) )
  
  #Create df for feature importance graph
  graph_df = pd.concat(objs = graph_dfs, axis = 0)
  if fnames is None:
    graph_df['Feature'] = ['X'+str(n) for n in graph_df['Feature_Num'].astype('int').values]
  else:
    graph_df['Feature'] = [fnames[n] for n in graph_df['Feature_Num'].astype('int').values]

  #Create df for pdp plot
  ns = [len(p) for p in pdp_x_list]
  if fnames is None:
    features = np.concatenate([['X'+str(i)]*n for i,n in zip(range(k), ns)])
  else:
    features = np.concatenate([[fnames[i]]*n for i,n in zip(range(k), ns)])

  feature_num = np.concatenate([[i]*n for i,n in zip(range(k), ns)])
  pdp_df = pd.DataFrame({'Feature': features, 'Feature_Num': feature_num, \
                         'X': np.concatenate(pdp_x_list), 'Y': np.concatenate(pdp_y_list)})

  ranked = graph_df.groupby('Feature',as_index = False)['Diff'].sum()
  ranked['rank'] = ranked['Diff'].rank(ascending = False)
    
  #Plot results
  print('Generating plot')
  feature_selector1 = alt.selection_single(on="mouseover", encodings=['y'])
  feature_selector1_click = alt.selection_multi(on="click", encodings=['y'])
  
  #Feature importance bar chart
  sort_order = ranked.sort_values('rank', ascending = True)['Feature'].values
  imp_xmin = graph_df.groupby('Feature')['Diff'].sum().min()
  imp_xmax = graph_df.groupby('Feature')['Diff'].sum().max()

  imp_chart = alt.Chart(title = 'Feature Importances').mark_bar(size=barsize).encode(
      x=alt.X('sum(Diff)', scale = alt.Scale(domain = (imp_xmin, imp_xmax)), axis = alt.Axis(title = 'Score Difference')),
      y=alt.Y('Feature',sort = sort_order),
      color = alt.value('#1f77b499')
      ).properties(
      width=w,
      height=h
      )

  imp_chart_hover = alt.Chart(title = 'Feature Importances').mark_bar(size=barsize).encode(
      x=alt.X('sum(Diff)', scale = alt.Scale(domain = (imp_xmin, imp_xmax)), axis = alt.Axis(title = 'Score Difference')),
      y=alt.Y('Feature',sort = sort_order),
      color = alt.condition(feature_selector1|feature_selector1_click, alt.value('#1f77b499'), alt.value('#FFFFFF00'))
      ).properties(
      width=w,
      height=h
      )

  imp_chart_layered = alt.layer(
        imp_chart,imp_chart_hover, data = graph_df
        ).add_selection(feature_selector1, feature_selector1_click)

  #PDP chart
  pdp_xmin = pdp_df['Y'].min()
  pdp_xmax = pdp_df['Y'].max()

  pdp_chart = alt.Chart(pdp_df, title = 'Partial Dep. Plots').mark_line().encode(
      alt.X('X'),
      alt.Y('Y', scale = alt.Scale(zero = False, domain = (pdp_xmin, pdp_xmax))),
      color=alt.Color('Feature:N',legend = None)
      ).properties(width=w,height=h)

  pdp_chart = pdp_chart.transform_filter(
      feature_selector1 | feature_selector1_click)

  final_plot = alt.hconcat(imp_chart_layered, pdp_chart).resolve_scale(color='independent'
  ).configure_axis(
      labelFontSize=fontsize,titleFontSize=fontsize
  ).configure_title(fontSize=fontsize
                    ).configure_view(stroke=None
                                     )

  return final_plot


def feature_importance_scores_v1(X, y, model, f, metric, higher_is_better = False,
                                 n_points = 20, predict_proba = False):

  #Choose a continuous feature, then make predictions for all imputed values of that feature
  if isinstance(X, pd.core.frame.DataFrame):
    is_df = True
    fnames = X.columns
  else:
    is_df = False

  c = cutpoints(X.iloc[:,f], n_points = n_points) if is_df == True else cutpoints(X[:,f], n_points = n_points)
  weights = c['weights'].values
  values = c['values'].values

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
  
  del X_impute
  
    #Create one-way PDP's by averaging across both axes
  f_not_pdp = np.average(pdp, axis = 0, weights = weights) 
  f_pdp = pdp.mean(axis=1).reshape(-1)
  f_pdp_interp = np.interp(X.iloc[:,f], values, f_pdp) if is_df == True else np.interp(X[:,f], values, f_pdp)

    #Center both pdps
  f_not_pdp = f_not_pdp - f_not_pdp.mean()
  f_pdp_interp = f_pdp_interp - f_pdp_interp.mean()

  #Calculate feature contributions to overall predictions
  yhat_uncentered = model.predict(X) if predict_proba == False else model.predict_proba(X)[:,1]
  yhat_mean = yhat_uncentered.mean()
  yhat = yhat_uncentered - yhat_mean

  #Calculate score differences
  if metric=='rmse':
    metric = rmse
  if metric=='mse':
    metric = mse

  #Compute feature importance scores (note that these scores use the un-centered PDPs)
  if higher_is_better == False:
    feature_score = metric(y, f_not_pdp + f_not_pdp.mean()) - metric(y, yhat_uncentered)
  else:
    feature_score = metric(y, yhat_uncentered) - metric(y, f_not_pdp + f_not_pdp.mean())

  return feature_score, values, f_pdp, f_not_pdp, f_pdp_interp
