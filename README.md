# RGID (Rapid Global Interpretability Dashboard): Fast, comprehensive, and interactive pattern discovery for black-box predictive models
This repository contains Python code and documentation for RGID (Rapid Global Interpretability Dashboard).

RGID is an interactive dashboard (built on the Altair visualization library) for exploring variable effects and interactions in machine learning models. 

RGID seeks to combine existing and novel interpretable ML visualizations, all in a single interactive dashboard that can be quickly produced from any regression or binary classification machine learning model with a .predict() (for regression) or .predict_proba() (for classification) method.

# Cite
Citaiton information will be added once the peer-review process concludes and the accompanying paper is published.

NEED TO ADD REPLICATION INSTRUCTIONS

# RGID Examples
Both RGID examples from the manuscript - i.e., from the primary simulation analysis and the SUD case study - are saved to the main repository directory as HTML files.

# Replication
The repository contains the Jupyter Notebook used for all analyses. Interested readers can replicate the primary simulation and benchmark analyses by running all code cells sequentially. Note that we include the already-run benchmark results in the 'data' directory, which can be loaded into the Jupyter notebook without needing to be re-computed. Finally, note that the SUD case study data and dashboard object are too large for Github's file size limitations. This issue will be resolved before publication of the paper.

# Documentation:
```python
run_rgid(X, y, model, metric,
             grid_size = 20,
             partial_out = True,
             h = 200,
             w = 200,
             fontsize=12,
             barsize = 10,
             feature_limit = None,
             predict_proba = False,
             pdp2_band_width = 0.10, 
             pdp3_band_width = 0.30,
             threeway_int_missing_threshold = 0.25,
             pdp_ips_trim_q = 0.9,
             interaction_quantiles = (0.25, 0.75),
             threeway_int_limit = 10,
             propensity_samples = 1000,
             feature_imp_njobs = 1,
             propensity_njobs = 4,
             threeway_int_njobs = 4,
             allow_hover = True):
``` 
Function arguments: 
 - X: Matrix or dataframe of predictor variables (typically a test set held out from model training)
 - y: Vector of outcomes. Currently, only regression or binary classification (but not multi-class classification) are supported. Binary outcomes should be formatted as 0/1.
 - metric: Accepts either 'rmse' (root mean squared error) or 'mse' (mean squared error) as string arguments, or a callable function that takes (y_true, y_predicted) as arguments and returns a score. Note that metrics should compute loss (not accuracy/performance), such that lower values are deemed better. We have included a negative AUC function (neg_auc) to use for binary classification evaluation.
 - grid_size: Number of grid points for evaluating each feature's partial dependence function.
 - h: Height for each of the six plots.
 - w: Width for each of the six plots.
 - fontsize: Chart font size.
 - barsize: Width of each bar in variable importance / H^2 plots.
 - feature_limit: Number of feature to consider. If not 'None', RGID will select the most important feature_limit features and only compute interactions among these features.
 - predict_proba: Set to TRUE for binary classification - this will instruct RGID to use the .predict_proba() method for the model class, rather than .predict().
 - pdp2_band_width: Quantile band width (denoted _b_ in the manuscript) for two-way PDP estimation.
 - pdp3_band_width: Quantile band width (denoted _b'_ in the manuscript) for three-way PDP estimation.
 - threeway_int_missing_threshold: Threshold for exluding a moderated PDP from RGID Plot 6. If more than this proportion of a three-way PDPs points are missing (i.e., because no observations have that particular combination of values for the three variables), the three-way interaction is given an H^2 score of zero. In practice, this prevents spurious H^2 scores from being assigned to interactions between mutually exclusive dummy variable categories.
 - pdp_ips_trim_q: Quantile for trimming propensity score weights when calculating two-way and three-way partila dependence functions.
 - interaction_quantiles: The quantiles above and below which interacting variables will be considered 'high' and 'low'. If set to the default value of (0.25, 0.75), then 'high' and 'low' values of interacting variables refer to the top quartile and bottom quartile, respectively.
 - threeway_int_limit: How many pairwise interactions to combine with each feature to form the set of considered three-way interactions. If set to the default value of 10, then the top 10 pairwise interactions will be interacted with each feature to produce the candidate set of three-way interactions.
 - propensity_samples: Number of samples to use for estimating propensity weights. When working with large datasets, setting this to a reasonable value smaller than the data's sample size can speed computation.
 - feature_imp_njobs: Number of parallel cores for feature importance and partial dependence function estimation. Set to 1 (sequential implementation) by default.
 - propensity_njobs: Number of parallel cores for estimating propensity scores. Set to 4 by default.
 - threeway_int_njobs: Number of parallel cores for threeway interaction score estimation. Set to 4 by default.
 - allow_hover: Set to False to have charts only filter when clicked (rather than hovered). 'False' is preferred when saving HTML copies of the chart.
