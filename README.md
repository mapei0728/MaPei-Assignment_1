# MaPei-Assignment_1

1. Data Cleaning and Missing Value Handling (0.Data_clean+NA.md)​
Handling Missing Values:
Remove high-missing columns and rows:
Drop columns with more than 50% missing values.
Drop rows with more than 40% missing values.
Final dataset size: (50441, 90) → (49331, 97) (after processing).

Standardization and Data Transformation:
Convert gender to binary: F → 0, M → 1.
Race Classification:
Standardize race categories (e.g., WHITE - BRAZILIAN mapped to WHITE).
Assign UNKNOWN to missing race values.
One-Hot Encoding for race variables.

Outlier Handling:
Detect and remove specific outliers (e.g., 999999 values).
Filter unrealistic weight values (20kg–300kg).
Remove extreme lab test values (e.g., baseexcess_min=-414).

Missing Value Imputation (KNN):
Compute optimal K = sqrt(n_samples) = 222.
Use KNNImputer for numerical variable imputation.
Save cleaned dataset (df_KNN-impute1.csv).

2. Correlation Analysis (1.Corr.md)​
Variable Classification:
Categorical Variables (e.g., gender, race).
Numerical Variables (e.g., admission_age, heart_rate_max).

Outlier Treatment:
Winsorization (3% upper/lower trimming to handle extreme values).

Correlation Analysis:
Compute feature correlation matrix (Pearson correlation).
Threshold set (>0.8), remove highly correlated variables.
24 highly correlated features removed (e.g., glucose_vital_mean, heart_rate_mean).

Final dataset size: (49331, 70).
Data Normalization:
Apply RobustScaler to normalize all numerical variables.


3. Elastic Net Feature Selection + Model Evaluation  (2.FS_L1-aki_stage.md)​
Data Import:
Load X+aki_stage-corr.csv (dataset after correlation filtering).
Convert target variable aki_stage to categorical type.

Elastic Net Feature Selection:
Train ElasticNet (l1_ratio=0.8) for sparse feature selection.
Select 23 key features, including:
admission_age, sbp_mean, spo2_mean, po2_max, etc.

Further Dimensionality Reduction:
Compute feature correlation matrix and remove highly correlated variables.
Final selected features: 16 (e.g., sbp_mean, po2_max, creatinine_min).

Data Splitting:
80/20 train-test split.

Model Evaluation:
Train Logistic Regression (SGDClassifier):
Accuracy: 44.96%, ROC AUC: 68.29%.
Train XGBoost:
Accuracy: 48.28%, ROC AUC: 70.50% (best-performing model).
Train Decision Tree (DT), Random Forest (RF), SVM for comparison.

4. Genetic Algorithm Feature Selection +  Model Evaluation  (3.FS_GA-aki_stage.ipynb)
Data Import
Load X+aki_stage-corr.csv (dataset after correlation filtering).
Convert target variable aki_stage to categorical type.

Genetic Algorithm Feature Selection
Implement Genetic Algorithm (GA) for feature selection using DEAP.

Fitness Function:
Use XGBoost to evaluate feature subsets based on ROC AUC score.

Selection, Crossover, and Mutation:
Selection: Tournament selection to retain top individuals.
Crossover: One-point crossover to create new feature subsets.
Mutation: Randomly flip bits to explore new feature combinations.

Further Dimensionality Reduction
Identify and retain only the top-ranked features from GA-selected feature subsets.
Compare the feature set obtained from GA with those selected by Elastic Net.

Data Splitting
80/20 train-test split for model training and evaluation.

Model Evaluation
Train XGBoost with GA-selected features:
Accuracy: Higher than Elastic Net model.
ROC AUC: Improved performance compared to Elastic Net.
Train Decision Tree (DT), Random Forest (RF), and SVM for comparison.
Evaluate models using Accuracy, Precision, Recall, F1 Score, and ROC AUC.
