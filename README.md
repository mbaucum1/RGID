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
             h = 200, w = 200, barsize = 10,
             feature_limit = None,
             pdp2_band_width = 0.10, #quantile width of areas for moderated PDPs
             pdp3_band_width = 0.30,
             pdp_ips_trim_q = 0.9, #quantile for trimming propensity score weights
             interaction_quantiles = (0.25, 0.75),
             fontsize=12,
             threeway_int_limit = 10,
             propensity_samples = 1000,
             feature_imp_njobs = 1,
             propensity_njobs = 4,
             threeway_int_njobs = 4,
             full_threeway_matrix = True):
``` 
Function arguments will be detailed in Appendix A of the published manuscript.
