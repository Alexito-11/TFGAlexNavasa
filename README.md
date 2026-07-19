# TFGAlexNavasa
TFG 2026 Alex Navasa


This repository contains the full pipeline developed for the classification of five cardiac pathologies from the ACDC dataset, combining a deep learning approach with a classical machine learning baseline built on handcrafted geometric descriptors.

The deep learning component includes data loading and preprocessing for cardiac MRI slices (data_loader.py), model configuration (config.py), and evaluation utilities for both slice-level and patient-level analysis, including majority-vote aggregation and confusion matrices (evaluation.py). Exploratory analysis of the image data, such as intensity distributions and class balance across training, validation, and test sets, is provided in data_analysis.py.

The classical machine learning component extracts geometric descriptors directly from the segmentation masks (quantification_methods.py) and compiles them into patient- and slice-level summary files (CSV). These descriptors are then used to train and compare several models, including Random Forest, XGBoost, SVM, MLP, KNN, and Logistic Regression, with feature importance assessed through permutation importance and SHAP values (modelsmlfeaturesgeometriques.py).

Finally, the repository includes correlation analyses linking the handcrafted geometric descriptors to the internal representations learned by the neural network, both at a univariate level (analisiscorrelaciounivariada.py) and a multivariate level using Random Forest regression (analisiscorrelaciomultivariada.py). Results are exported as CSV files and heatmap or bar chart visualizations, organized in dedicated output folders.
