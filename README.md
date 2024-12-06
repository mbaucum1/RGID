# RGID (Rapid Global Interpretability Dashboard): Fast, comprehensive, and interactive pattern discovery for black-box predictive models
This repository contains Python code and documentation for RGID (Rapid Global Interpretability Dashboard).

RGID is an interactive dashboard (built on the Altair visualization library) for exploring variable effects and interactions in machine learning models. 

RGID seeks to combine existing and novel interpretable ML visualizations, all in a single interactive dashboard that can be quickly produced from any regression or binary classification machine learning model with a .predict() (for regression) or .predict_proba() (for classification) method.

# Cite
Citaiton information will be added once the peer-review process concludes and the accompanying paper is published.

NEED TO ADD REPLICATION INSTRUCTIONS

# RGID Examples
Both RGID examples from the manuscript - i.e., from the primary simulation analysis and the SUD case study - are saved to the 'results' directory as HTML files.

# Replication
The repository contains the Jupyter Notebook used for all analyses. Interested readers can replicate the primary simulation and benchmark analyses by running all code cells sequentially. Note that we include the already-run benchmark results in the 'data' directory, which can be loaded into the Jupyter notebook without needing to be re-computed. Finally, note that the SUD case study data and dashboard object are too large for Github's file size limitations. This issue will be resolved before publication of the paper.

# Documentation:
```python
run_rgid(X, y, model, metric,
             predict_method = None,
             grid_size = 20,
             h = 200,
             w = 200,
             barsize = 10,
             fontsize=12,
             feature_limit = None,
             pdp2_band_width = 0.10, 
             pdp3_band_width = 0.30,
             pdp_ips_trim_q = 0.9, 
             interaction_quantiles = (0.25, 0.75),
             threeway_int_limit = 10,
             propensity_samples = 1000,
             feature_imp_njobs = 1,
             propensity_njobs = 4,
             threeway_int_njobs = 4,
             full_threeway_matrix = True,
             pdp_legend = False):
``` 
Function arguments:
 - X: Tabular dataset
 - y: Continuous or binary outcome
 - Model: Trained model object
 - Metric: Loss function: Accepts 'rmse', 'mae', or a callable function.
 - predict_method: If the outcome vector has more than two unique values, RGID will assume it was passed a regression model and will use the model's predict method in all calculations; otherwise, it will use predict_proba, which generates classification probabilities for most commonly-used Python model classes. Setting predict_method to True will override this behavior and use the predict method, even for classification models.
 - grid_size: The number of grid points to use when calculating PD functions.
 - w, h: Width and height, respectively, of each RGID plot (in pixels).
 - barsize: Size of bars, in pixels
 - fontsize (default: 12): Font size for all plot axes and labels.
 - feature_limit (default: None): Limits all RGID plots to only the top feature_limit most important features.
 - pdp2_band_width (default: 0.10): The width of the quantile band used to calculate pairwise moderated PD functions in Plot 4.
 - pdp3_band_width (default: 0.30): The width of the quantile band used to calculate three-way moderated PD functions in Plot 6.
 - pdp_ips_trim_q (default: 0.9): The quantile at which to clip propensity weights for calculating moderated PD functions
 - interaction_quantiles (default: (0.25, 0.75)): A tuple of quantiles that define low' andhigh' values of moderated variables in pairwise and three-way PD functions
 - threeway_int_limit (default: 10): Number of three-way interactions to test for each feature.
 - propensity_samples (default: 1000): Number of randomly-selected observations on which to fit propensity weights for moderated PD functions; can speed up computation in large datasets.
 - feature_imp_njobs (default: 1): Number of cores to use for PD-AVID (passed to joblib's Parallel function).
 - propensity_njobs (default: 4): Number of cores to use for moderated PD function propensity scores.
 - threeway_int_njobs (default: 4): Number of cores to use for three-way H^2 calculation.
 - full_threeway_matrix (default: True): If set to True (default), Plot 5 will appear as a full matrix-style heatmap showing all tested three-way interactions with the focal variable selected in Plot 1. If False, Plot 5 will appear as a simple bar chart showing the strongest three-way interaction for each variable.
 - pdp_legend (default: False): Whether to produce a legend of variable names in Plot 2 (top-right, visualizing PDPs). 
 

