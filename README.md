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
