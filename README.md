# Parkinson's Disease Detection from Voice Recordings

Binary classification (healthy vs Parkinson's) from voice-derived features, evaluated in a way that keeps all recordings from one person in the same fold.

## Dataset

- **Source:** [UCI Machine Learning Repository, Parkinson's Data Set](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Size:** 195 voice recordings from 32 people (24 with Parkinson's disease, 8 healthy), about six recordings per person
- **Features:** 22 voice measurements (for example jitter, shimmer, HNR, RPDE, DFA, PPE)
- **Target:** `status` (1 = Parkinson's, 0 = healthy)
- **Class balance:** 147 of 195 recordings (75.4%) are from people with Parkinson's, so always predicting "Parkinson's" already gives an accuracy of 0.754
- **Note:** Each person contributes about six recordings, so a random train/test split can put the same person in both sets and overstate performance. This project therefore also reports subject-aware cross-validation.

## Methods

- **Preprocessing:** median imputation and standard scaling inside a scikit-learn pipeline
- **Models:** Random Forest (300 trees) and Logistic Regression
- **Evaluation:** subject-aware 5-fold cross-validation (`StratifiedGroupKFold`), random 5-fold cross-validation, and a single 80/20 split (`random_state=42`)
- **Metrics:** accuracy, F1, ROC AUC

## Results

Baseline accuracy (always predicting Parkinson's): 0.754. Values are mean ± standard deviation across folds.

| Model | Evaluation | Accuracy | F1 | ROC AUC |
|---|---|---|---|---|
| Random Forest | Subject-aware 5-fold CV | 0.786 ± 0.064 | 0.864 ± 0.051 | 0.807 ± 0.034 |
| Logistic Regression | Subject-aware 5-fold CV | 0.787 ± 0.082 | 0.865 ± 0.050 | 0.736 ± 0.169 |
| Random Forest | Random 5-fold CV | 0.892 ± 0.066 | 0.931 ± 0.041 | 0.958 ± 0.030 |
| Logistic Regression | Random 5-fold CV | 0.867 ± 0.050 | 0.913 ± 0.032 | 0.905 ± 0.042 |
| Random Forest | Single 80/20 split (39 test recordings) | 0.949 | 0.970 | 0.933 |
| Logistic Regression | Single 80/20 split | 0.897 | 0.941 | 0.835 |

Random splits give higher scores because the same person appears in both training and test data. With subject-aware evaluation, accuracy is about 0.79, only a little above the baseline, so these voice features alone separate the groups only moderately on this small dataset.

## Limitations

- Only 32 people, so results vary between folds and random seeds
- Voice features only; this is a learning project, not a diagnostic tool

## How to run

```bash
git clone https://github.com/Tugceaydinn/parkinsonsdisease.git
cd parkinsonsdisease
pip install -r requirements.txt
jupyter notebook pd.ipynb
python eval_pd.py data/parkinsons.csv
```

Running the notebook creates `results/classification_metrics.json`, `results/feature_importance.png` and `results/classification_pipeline.joblib`.

## Project structure

- `data/parkinsons.csv`: the dataset
- `pd.ipynb`: preprocessing, Random Forest model, feature importance
- `eval_pd.py`: model comparison with random and subject-aware cross-validation
- `results/`: generated metrics and plots
- `requirements.txt`: dependencies

## References

- Little, M. A., et al. (2007). Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection. BioMedical Engineering OnLine. (Check the exact reference on the UCI page.)

## Future work

- MLP baseline in PyTorch using the same subject-aware folds
- Feature selection and SHAP analysis
- Regression of `total_UPDRS` on the separate UCI Parkinson's Telemonitoring dataset
