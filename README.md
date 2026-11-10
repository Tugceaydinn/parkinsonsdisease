# Parkinson’s Disease – UPDRS Regression & Classification

This project focuses on **predicting and classifying Parkinson’s disease severity** using voice-derived features and machine learning models.  
It includes both **regression** (predicting `total_UPDRS`) and **binary classification** (healthy vs patient) pipelines.

---

## Project Overview

- **Objective:** Predict UPDRS scores and classify Parkinson's severity  
- **Dataset:** Parkinson’s dataset (e.g. from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons))  
- **Tech Stack:** Python, pandas, scikit-learn, matplotlib  
- **Methods:**  
  - Data preprocessing  
  - **Regression:** SVR, Lasso, KNN, Random Forest  
  - **Classification:** Logistic Regression, Random Forest  
  - ROC analysis & standard performance metrics

---

## Dataset

- **Source:** [UCI Machine Learning Repository, Parkinson's Data Set](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Size:** 195 voice recordings from 32 people (24 with Parkinson's disease, 8 healthy), about six recordings per person
- **Features:** 22 voice measurements (for example jitter, shimmer, HNR, RPDE, DFA, PPE)
- **Target:** `status` (1 = Parkinson's, 0 = healthy)
- **Class balance:** 147 of the 195 recordings (75.4%) are from people with Parkinson's, so always predicting "Parkinson's" already gives an accuracy of 0.754. Results should be compared with this baseline.
- **Important:** Because each person contributes about six recordings, a random train/test split can put the same person in both sets and overstate performance. This project therefore also reports subject-aware cross-validation, where all recordings from one person stay in the same fold.
- **Citation:** Little, M. A., et al. (2007). *Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection.* BioMedical Engineering OnLine. Check the exact reference on the UCI page before you commit.

## Project Structure

parkinson-s-disease/
├─ data/ # dataset goes here if shareable
│ └─ README.md # link/instructions if data not included
├─ results/ # generated graphs & metrics
│ └─ .gitkeep
├─ pd.ipynb # main notebook (regression + classification)
├─ requirements.txt # dependencies
└─ README.md



> `.gitkeep` is only used to keep the results folder visible in the repo.

---

##  How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/Tugceaydinn/parkinson-s-disase.git
   cd parkinson-s-disase
Install dependencies


pip install -r requirements.txt
Place the dataset

Put parkinsons.csv inside the data/ folder.

The notebook uses:


DATA_PATH = 'data/parkinsons.csv'
Run the notebook

jupyter notebook "pd.ipynb"
Outputs

results/roc_curve.png – ROC curve plot

results/regression_test_mse.csv – regression results

results/classification_metrics.csv – classification metrics

 Model Performance (Example)
Model	Type	Metric	Score
SVR	Regression	MSE ↓	91.71
Random Forest	Regression	MSE ↓	184.66
Logistic Regression	Classification	Acc ↑	0.87
Random Forest	Classification	Acc ↑	0.91

 In our tests, SVR achieved the lowest error for regression, and Random Forest provided the best classification accuracy.

Visualizations

 References
UCI Parkinson’s Disease Dataset

Little MA, McSharry PE, Hunter EJ, Spielman J, Ramig LO (2009).

 Author
Tuğçe Aydın
MSc Data & Computational Science @ UCD
Dublin, Ireland
LinkedIn · GitHub

Future Work
Hyperparameter tuning

Exploring MLP / deep learning baselines

Feature selection and SHAP analysis

Real-time voice detection prototype

