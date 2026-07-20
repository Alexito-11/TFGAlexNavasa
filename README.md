# Study of the intermediate representations of a neural network for cardiac disease prediction

Bachelor's thesis by Àlex Navasa.

## Repository structure

    TFGAlexNavasa/
    -CSV/
    -analisisfeaturesmanuals/
    -correlació univariada/
    -correlació multivariada/
    -xarxaResnet18/
    -xarxaentrenadadesde0/

# Data setup

This project needs the ACDC dataset to run: both neural
network folders (xarxaResnet18 and xarxaentrenadadesde0) load the raw images directly. The
dataset is not included in this repository, so you need to download it from
https://acdc.creatis.insa-lyon.fr/ and unzip it keeping the original folder structure
(database/training/ and database/testing/, with one folder per patient).

The scripts point to the data through absolute paths, so before running anything you have
to set them to your own location: BASE_DIR_TRAIN and BASE_DIR_TEST in
quantification_methods.py, and the equivalent paths in the config.py of each network
folder.

modelsmlfeaturesgeometriques.py does not read the images: it works on summary_training.csv
and summary_testing.csv, which are produced by quantification_methods.py, so that one has
to be run first.

# Requirements

- Python 3.x
- PyTorch / torchvision (for the CNN modules)
- pandas, numpy, scikit-learn (for the correlation analysis and the manual features)
- matplotlib / seaborn (for the heatmaps and plots)

# CSV

Generation of the training and test datasets used to train the neural networks.

Usage: run the generator script to create the training/testing CSV files before training
any model.

# analisisfeaturesmanuals

Analysis of geometric features extracted manually from the images, and their
quantification using classical Machine Learning models.

- quantification_methods.py: quantification methods for the manual descriptors/features;
  reads the ACDC images and writes summary_training.csv and summary_testing.csv.
- modelsmlfeaturesgeometriques.py: defines and trains ML models (non-neural) on those
  geometric features.

Usage:

    python quantification_methods.py
    python modelsmlfeaturesgeometriques.py

# correlació univariada

Univariate correlation analysis between the features and the target variables, split by
cardiac phase (ED/ES) and aggregation level (per patient / per slice).

- analisiscorrelaciounivariada.py: main script; computes the correlation matrices,
  generates the heatmaps and extracts the top correlations.

Generated outputs:

- *_correlation_matrix.csv: full correlation matrix
- *_heatmap.png: visualisation of the correlation matrix
- *_top_correlations.csv: list of the most relevant correlations

These are generated for training/testing, ED/ES, and per patient/slice aggregation.

Usage:

    python analisiscorrelaciounivariada.py

# correlació multivariada

Multivariate correlation analysis (multiple regression, R2) between sets of features and
the target variables.

- analisiscorrelaciomultivariada.py: main script; fits the multivariate models and
  computes the R2 coefficient.

Generated outputs:

- *_multivariate_R2.csv: numerical R2 results per model
- *_multivariate_R2.png: visualisation of the results

These are generated for training/testing, ED/ES, and per patient/slice aggregation.

Usage:

    python analisiscorrelaciomultivariada.py

# xarxaResnet18

CNN based on the ResNet18 architecture (transfer learning / feature extraction), used for
image feature extraction and biomarker prediction.

- config.py: model parameters (hyperparameters, paths, etc.).
- data_loader.py: loading and preprocessing of the image data.
- data_analysis.py: exploratory analysis of the input data.
- model_resnet.py: definition of the ResNet18-based FeatureExtractor.
- train_utils.py: helper functions for training.
- trainer.py: training logic.
- evaluation.py: evaluation of the trained model on the test set.
- main_resnet.py: entry point, runs the full pipeline (loading, training, evaluation).

Usage:

    python main_resnet.py

# xarxaentrenadadesde0

CNN trained from scratch (without transfer learning), as a comparison against the ResNet18
approach.

- config.py: model parameters (hyperparameters, paths, etc.).
- dataloaderxarxa0.py: loading and preprocessing of the image data.
- modelxarxa0.py: definition of the CNN architecture trained from scratch.
- train_utils.py: helper functions for training.
- trainerxarxa0.py: TrainerCNN class, containing the training logic.
- evaluationxarxa0.py: evaluation of the trained model on the test set.
- mainxarxa0.py: entry point, runs the full pipeline (loading, training, evaluation).

Usage:

    python mainxarxa0.py

Note: config.py and train_utils.py are the same files in both network folders
(xarxaResnet18 and xarxaentrenadadesde0); they are duplicated because their contents are
shared by both models.

# Author

Àlex Navasa — Bachelor's thesis

