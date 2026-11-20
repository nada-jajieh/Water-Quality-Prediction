# Water Quality Prediction Using Machine Learning

A machine learning classification project that predicts whether water is **potable (safe to drink)** or **not potable** based on measured water-quality characteristics.

## Project Overview

The goal is to compare different supervised classification algorithms for predicting water potability. The project includes exploratory data analysis, missing-value handling, model training, hyperparameter tuning, and evaluation.

## Dataset

The project uses `water_potability.csv`, containing **3,276 samples**, **9 numerical input features**, and one target column (`Potability`).

| Target value | Meaning | Samples |
|---|---|---:|
| `0` | Not potable | 1,998 |
| `1` | Potable | 1,278 |

**Input features:** `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, and `Turbidity`.

Some records have missing values, particularly in `ph`, `Sulfate`, and `Trihalomethanes`.

## Workflow

1. **Exploratory Data Analysis (EDA):** inspect features, class distribution, and missing values.
2. **Preprocessing:** experiment with mean and median imputation for missing values. For unbiased evaluation, fit the imputer on training data only and apply it to validation/test data.
3. **Model Training:** train and compare three classification algorithms.
4. **Validation:** use 5-fold stratified cross-validation and a hold-out test set.
5. **Hyperparameter Tuning:** search for improved model configurations.
6. **Evaluation:** compare models using classification metrics and confusion matrices; inspect signs of overfitting.

## Machine Learning Models

- **Logistic Regression** — a linear classification baseline.
- **Random Forest** — an ensemble of decision trees.
- **XGBoost** — a gradient-boosted tree ensemble.

## Evaluation Metrics

- **Accuracy** — proportion of correct predictions.
- **Precision** — proportion of predictions for a chosen class that are correct.
- **Recall** — proportion of actual samples of a chosen class correctly identified.
- **F1-score** — harmonic mean of precision and recall.
- **Confusion Matrix** — counts of correct and incorrect predictions for each class.

Because wrongly predicting unsafe water as safe can have serious consequences, the project also examines the classification performance for **class `0` (not potable)** rather than relying on accuracy alone.

## Results

In the hyperparameter-tuning experiment, the following best cross-validation accuracies were observed:

| Model | Best CV accuracy |
|---|---:|
| Logistic Regression | 60.99% |
| Random Forest | **68.21%** |

Random Forest performed best among these reported results. A separate overfitting check recorded **98.28% training accuracy** and **66.92% testing accuracy**, highlighting the importance of comparing training and unseen-data performance. See the notebook for all experimental runs and results, including XGBoost.

> Results depend on the experimental setup and should be interpreted alongside class-specific precision, recall, F1-score, and the confusion matrix.

## Technologies

- Python
- Jupyter Notebook
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- XGBoost

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Water-Quality-Prediction.git
   cd Water-Quality-Prediction
   ```
2. Install the main dependencies:
   ```bash
   pip install notebook pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
3. Place `water_potability.csv` in the location expected by the notebook.
4. Start Jupyter:
   ```bash
   jupyter notebook
   ```
5. Open `Water_Potability.ipynb` (or the notebook filename in this repository) and run its cells in order.

## Repository Structure

```text
Water-Quality-Prediction/
├── Water_Potability.ipynb    # Main analysis and ML experiments
├── water_potability.csv      # Dataset (if redistribution is permitted)
└── README.md                # Project documentation
```

## Notes

- Predictions from this dataset are **not** a substitute for laboratory testing or regulatory certification of drinking water.
- If the dataset is distributed separately, add its original source and license information before sharing the repository.
