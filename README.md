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

# Documentation:
```
dive_dashboard_v1(X, y, model, metric, higher_is_better = False, predict_proba = False, pdp_n_points = 20, h = 200, w = 200, barsize = 10, fontsize=12):
``` 
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
