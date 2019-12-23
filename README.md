# Using Machine Learning to identify Hard Bounce E-mails

This project explore the Classification techniques in Machine Learning to train and identify hard bounce e-mails.

## Folders

*datasets*: compressed dataset used to train the algorithm
*functions*: python functions built to reutilize during the code
*models*: models saved after trained

## Scripts

*01_feature_engineering.ipynb*: Feature engineering over original dataset
*02_exploratory_analysis.ipynb*: Exploratory analysis after feature engineering
*03_0_train_extremeGradientBoosting.ipynb*: Train Extreme Gradient Boost algorithm using the oversampled dataset
*03_1_train_extremeGradientBoosting_WithImbalance.ipynb*: Train Extreme Gradient Boost algorithm using the imbalanced dataset
*03_1_train_randomForest.ipynb*: Train Random Forest algorithm using the oversampled dataset
*04_0_apply_extremeGradientBoosting.ipynb*: Apply trained Extreme Gradient Boost on the imbalanced dataset
*04_1_apply_randomForest.ipynb*: Apply trained Random Forest on the imbalanced dataset

