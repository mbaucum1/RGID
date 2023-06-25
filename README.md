# DIVE
Python code and documentation for DIVE - Dashboard for Interpretable Variable Explanations.

DIVE is an interactive dashboard (built on the Altair visualization library) for exploring variable effects and interactions in machine learning models. Code originally presented at the 2023 INFORMS Business Analytics Conference in Aurora, CO.

DIVE seeks to combine existing and novel interpretable ML visualizations, all in a single interactive dashboard that can be quickly produced from any scikit-learn or keras machine learning model. Planned dashboard components include:
 - Variable importance scores (decomposed into linear, nonlinear, and interaction components)
 - Partial dependence (PD) plots and accumulated local effects (ALE) plots
 - Variable importance scores and PD plots for pairwise interactions (with considerable runtime advantages over existing packages)
 - Estimated model performance from user-defined subsets of features

Upcoming page edits:
 - 4/18/2023: Limited code for generating interactive feature importance and partial dependence (PD) plots.
 - Summer 2023: Additional algorithms for pairwise interaction estimation
 - Fall 2023: Allow users to see model performance under customized subsets of features
 - Fall 2023: Clustering for identifying 'sub-models' that can be applied to homogenous portions of training data

# RGID Example
The [rgid_example] HTML file can be opened in any web browser, and visualizes an interactive RGID example using the simulated dataset described in the manuscript. Note that, due to file space limitations, hover actions are disabled in the example file. When run in a Jupyter Notebook, hovering over a given variable in Plot 1 will filter all other plots to that focal variable, and hovering over a variable in Plots 3 or 5 will filter Plots 4 and 6 (respectively) to that variable.

# Documentation:
```python
dive_dashboard_v1(X, y, model, metric, higher_is_better = False, predict_proba = False, pdp_n_points = 20, h = 200, w = 200, barsize = 10, fontsize=12):
``` 
Function arguments: 

 - X: Matrix or dataframe of predictor variables. 
 - y: Vector of outcomes. Currently, only regression or binary classification (but not multi-class classification) are supported. Binary outcomes should be formatted as 0/1.
 - metric: Accepts either 'rmse' (root mean squared error) or 'mse' (mean squared error) as string arguments, or a callable function that takes (y_true, y_predicted) as arguments and returns a score.
 - higher_is_better (default = False): Whether higher values on 'metric' indicate better model performance. Set to 'True' if using metrics such as accuracy, AUC, etc.
 - pdp_n_points (default = 20).: Number of points for evaluting univariate partial dependence plots (PDPs).
 - predict_proba (default = False): Set to 'True' to generate prediction's from the 'predict_proba()' method of 'model'. If left 'False', function will use model's 'predict' method (which may generate binary outputs rather than probability predictions, in the case of classifiers).
 - h (default = 200): Height of each plot.
 - w (default = 200): Width of each plot.
 - barsize (default = 10): Default width of bars.
 - fontsize (default = 12): Default plot font size.

# Example
```python
#Packages required for dive_dashboard_v1 to work:
import numpy as np
import pandas as pd
import altair as alt
from altair import datum
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Loading in data: Telecom churn dataset from https://archive.ics.uci.edu/ml/datasets/Iranian+Churn+Dataset
churn = pd.read_csv('Customer Churn.csv')
churn.columns = ['Call Failure','Complaint','Subscription Length','Charge Amount','Seconds of Use','Frequency of Use',
                 'Frequency of SMS','Distinct Called Numbers','Age Group','Tariff Plan','Active/Inactive','Age Num','Customer Value',
                 'FN','FP','Churn']
churn['Active/Inactive'] = churn['Active/Inactive'] - 1

#Split into train and test sets
from sklearn.model_selection import train_test_split
xvars = [c for c in churn.columns if c not in ['Age Group','Seconds of Use','FN','FP','Churn']]
X = churn[xvars]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, churn['Churn'], test_size = 0.3)

#Train random forest predicting churn
from sklearn.ensemble import RandomForestClassifier
n_est = 500; depth = 10
rf=RandomForestClassifier(max_features='sqrt',verbose=0,n_jobs=-1,max_samples=0.2,max_depth=depth,n_estimators=n_est, random_state=2030)
rf.fit(X=X_train_c, y = y_train_c)

#Create DIVE dashboard
from sklearn.metrics import roc_auc_score
churn_model_dashboard = dive_dashboard_v1(X = X_test_c, y = y_test_c, model = rf,
                            metric = roc_auc_score, higher_is_better = True,
                            predict_proba = True, pdp_n_points = 20,
                            h = 200, w = 200, barsize = 10,fontsize=12)

churn_model_dashboard
```
![DIVE example](https://github.com/mbaucum1/DIVE/blob/main/dive_example.png?raw=true)
