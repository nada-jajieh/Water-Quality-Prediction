#!/usr/bin/env python
# coding: utf-8

# # Water Quality Prediction Using Machine Learning
# 
# This project aims to predict whether water is safe for drinking based on its physicochemical properties using Machine Learning algorithms. Three classification algorithms will be trained and evaluated to compare their performance and identify the most effective model for water potability prediction.

# # 1. Importing library

# **Purpose:** Import the libraries needed for data analysis, visualization, preprocessing, model training, and evaluation.

# In[2]:


# 1. Data Manipulation & Analysis
import pandas as pd
import numpy as np
# 2. Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# 3. Statistical Analysis
from scipy import stats
# 4. Data Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
# 5. Data Splitting & Cross Validation
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV
)
# 6. Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# 7. Model Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
# 8. Feature Importance
from sklearn.inspection import permutation_importance
# 9. Save & Load Model
import joblib


# # 2.Loading data 

# **Purpose:** Load the water quality dataset into a Pandas DataFrame for analysis.

# In[3]:


df = pd.read_csv("Datasets/Water Quality Prediction/water_potability.csv")


# ## View top 5 row

# In[7]:


df.head()


# # 3.Exploratory Data Analysis (EDA)

# ## Dataset detail

# **What we check:** The dataset has 3,276 samples, 9 input features, and 1 target column (`Potability`). `df.info()` shows data types and non-missing counts.

# In[8]:


df.info()


# ## check balance

# **Class distribution:** `0 = Not Potable` (60.99%) and `1 = Potable` (39.01%). The classes are not evenly balanced.

# In[9]:


# Number of samples in each class
print("Class Counts:")
print(df["Potability"].value_counts(dropna=False))

# Percentage of each class
print("\nClass Percentages:")
print(df["Potability"].value_counts(normalize=True, dropna=False) * 100)


# ## check missing value

# **Missing data:** 1,265 rows contain at least one missing value. The affected features are `ph`, `Sulfate`, and `Trihalomethanes`.

# In[4]:


print("Dataset Shape:", df.shape)
print("Rows with missing values:", df.isnull().any(axis=1).sum())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMissing Percentage:")
print((df.isnull().mean() * 100).round(2))


# ## Outliers detection

# **Outliers:** The IQR method flags unusually high or low values in each feature. This step detects and counts outliers; it does not remove them.

# In[12]:


# Outlier Detection using IQR

features = df.drop(columns=["Potability"])

Q1 = features.quantile(0.25)
Q3 = features.quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = (
    (features < lower_bound) |
    (features > upper_bound)
)

outlier_summary = pd.DataFrame({
    "Outlier Count": outliers.sum(),
    "Outlier Percentage": (
        outliers.mean() * 100
    ).round(2)
})

print(outlier_summary)


# ## Correlation matrix

# **Correlation:** Pearson correlation measures the linear relationship between numerical variables. The heatmap makes these relationships easier to inspect.

# In[13]:


corr_matrix = df.corr(method="pearson")

print("Correlation Matrix:")
print(corr_matrix.round(3))


# In[14]:


#correlation heatmap(visualisation )

plt.figure(figsize=(12, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Heatmap - Water Quality")
plt.tight_layout()
plt.show()


# # 4. Data preprocessing

# ## Train/Test dataset 
# using 5 fold cross validation method

# **5-Fold Cross-Validation:** Split the dataset into 5 folds. Train on 4 folds and validate on the remaining fold, repeating 5 times. `StratifiedKFold` maintains similar class proportions in each split.

# In[6]:


from sklearn.model_selection import StratifiedKFold

# Separate Features and Target
X = df.drop(columns=["Potability"])
y = df["Potability"]

# Initialize Stratified 5-Fold Cross-Validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Save folds for later use
folds = list(skf.split(X, y))

# Display each fold
for fold, (train_idx, val_idx) in enumerate(folds, start=1):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    print(f"\n========== Fold {fold} ==========")

    print("Training Samples:", len(X_train))
    print("Validation Samples:", len(X_val))

    print("\nTraining Class Distribution (%):")
    print((y_train.value_counts(normalize=True) * 100).round(2))

    print("\nValidation Class Distribution (%):")
    print((y_val.value_counts(normalize=True) * 100).round(2))


# ## fill missing value

# **Median imputation:** Replace missing values with the median of each feature. The median is calculated from the training fold only, then applied to its validation fold to avoid data leakage.

# In[7]:


processed_folds = []

for iteration, (train_idx, val_idx) in enumerate(folds, start=1):

    # Get training and validation data
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()

    # Median Imputation
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index
    )

    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val),
        columns=X.columns,
        index=X_val.index
    )

    # Save processed data
    processed_folds.append({
        "iteration": iteration,
        "X_train": X_train_imputed,
        "X_val": X_val_imputed,
        "y_train": y_train,
        "y_val": y_val
    })

    print(f"\n--- Iteration {iteration} ---")
    print("Train Missing:",
          X_train_imputed.isnull().sum().sum())
    print("Validation Missing:",
          X_val_imputed.isnull().sum().sum())

    print("Medians used:")
    medians = pd.Series(
        imputer.statistics_, index=X.columns
    )
    print(medians[
        ["ph", "Sulfate", "Trihalomethanes"]
    ].round(3))


# ## feature scaling 

# **Feature scaling:** `StandardScaler` standardizes input features using the training fold’s mean and standard deviation. It is applied to Logistic Regression in this experiment.

# In[8]:


scaled_iterations = []

for iteration, data in enumerate(processed_folds, start=1):

    X_train = data["X_train"]
    X_val = data["X_val"]

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit and transform training data
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    # Transform validation using training statistics
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )

    # Save scaled data for Logistic Regression
    scaled_iterations.append({
        "iteration": iteration,
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "y_train": data["y_train"],
        "y_val": data["y_val"]
    })

    print(f"\n--- Iteration {iteration} ---")

    print("Training Mean:")
    print(X_train_scaled.mean().round(3))

    print("\nTraining Standard Deviation:")
    print(X_train_scaled.std(ddof=0).round(3))


# ## starting training 

# ## 5 fold cross validation
# **Model training:** Compare Logistic Regression, Random Forest, and XGBoost on all 5 folds. Report Accuracy, Precision, Recall, F1, and ROC-AUC, plus the number of not-potable samples incorrectly predicted as potable.

# In[9]:


# Store results for all models
results = []

# Define algorithms
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        eval_metric="logloss",
        random_state=42
    )
}

# Train each algorithm across 5 iterations
for model_name, model_template in models.items():

    print(f"\n===== {model_name} =====")

    # Use scaled data for Logistic Regression
    if model_name == "Logistic Regression":
        data_folds = scaled_iterations
    else:
        data_folds = processed_folds

    for iteration, data in enumerate(data_folds, start=1):

        X_train = data["X_train"]
        X_val = data["X_val"]
        y_train = data["y_train"]
        y_val = data["y_val"]

        # Create a fresh model for each iteration
        from sklearn.base import clone
        model = clone(model_template)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(
            y_val, y_pred, zero_division=0
        )
        recall = recall_score(
            y_val, y_pred, zero_division=0
        )
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob)

        # Actual 0, predicted 1 = False Potable
        tn, fp, fn, tp = confusion_matrix(
            y_val, y_pred, labels=[0, 1]
        ).ravel()

        # Save results
        results.append({
            "Algorithm": model_name,
            "Iteration": iteration,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": auc,
            "False_Potable": fp
        })

        print(
            f"Iteration {iteration}: "
            f"Accuracy={accuracy:.3f}, "
            f"F1={f1:.3f}, "
            f"False Potable={fp}"
        )

# Results for all 15 runs
results_df = pd.DataFrame(results)

print("\n===== All Results =====")
print(results_df.round(3).to_string(index=False))


# In[10]:


metrics = [
    "Accuracy", "Precision",
    "Recall", "F1", "ROC_AUC"
]

# Mean performance
mean_results = results_df.groupby("Algorithm")[metrics].mean()

# Standard deviation across iterations
std_results = results_df.groupby("Algorithm")[metrics].std()

# Total false potable classifications
false_potable = results_df.groupby(
    "Algorithm"
)["False_Potable"].sum()

comparison = (mean_results * 100).round(2)
comparison["Total_False_Potable"] = false_potable

print("===== Model Comparison =====")
print(comparison.sort_values(
    "F1", ascending=False
).to_string())

print("\n===== Standard Deviation (%) =====")
print((std_results * 100).round(2).to_string())


# ## confusion matrix

# **Confusion matrices:** Compare actual and predicted classes for each model across the 5 validation folds. The displayed labels are ordered **Potable (1)**, then **Not Potable (0)**; the bottom-right cell counts correctly identified not-potable water. **Note:** Precision, Recall, and F1 in the earlier model comparison use the default positive class **1 (Potable)**.

# In[11]:


tn, fp, fn, tp = confusion_matrix(
    y_val, y_pred, labels=[1, 0]
).ravel()


# In[12]:


fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (model_name, model_template) in zip(axes, models.items()):

    data_folds = (
        scaled_iterations
        if model_name == "Logistic Regression"
        else processed_folds
    )

    all_true = []
    all_pred = []

    for data in data_folds:
        model = clone(model_template)

        model.fit(data["X_train"], data["y_train"])
        y_pred = model.predict(data["X_val"])

        all_true.extend(data["y_val"])
        all_pred.extend(y_pred)

    # TP = Not Potable (class 0)
    cm = confusion_matrix(
        all_true, all_pred, labels=[1, 0]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Potable", "Not Potable"]
    )

    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(model_name)

plt.tight_layout()
plt.show()


# ## hold out method

# **Hold-Out evaluation:** Use 70% of the data for training and 30% for testing. Fit imputation and scaling on the training set only, then evaluate the three models on the test set.

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Separate Features and Target
X = df.drop(columns=["Potability"])
y = df["Potability"]

# Hold-Out: 70% Training, 30% Testing
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

print("Training Samples:", len(X_train_h))
print("Testing Samples:", len(X_test_h))

# Fill missing values using TRAIN median only
imputer_h = SimpleImputer(strategy="median")

X_train_h = pd.DataFrame(
    imputer_h.fit_transform(X_train_h),
    columns=X.columns,
    index=X_train_h.index
)

X_test_h = pd.DataFrame(
    imputer_h.transform(X_test_h),
    columns=X.columns,
    index=X_test_h.index
)

print("Train Missing:", X_train_h.isnull().sum().sum())
print("Test Missing:", X_test_h.isnull().sum().sum())

# Feature Scaling for Logistic Regression
scaler_h = StandardScaler()

X_train_scaled_h = scaler_h.fit_transform(X_train_h)
X_test_scaled_h = scaler_h.transform(X_test_h)


# In[14]:


from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import pandas as pd

holdout_results = []

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (model_name, model_template) in zip(axes, models.items()):

    model = clone(model_template)

    # Use scaling only for Logistic Regression
    if model_name == "Logistic Regression":
        X_tr = X_train_scaled_h
        X_te = X_test_scaled_h
    else:
        X_tr = X_train_h
        X_te = X_test_h

    model.fit(X_tr, y_train_h)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    holdout_results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy_score(y_test_h, y_pred),
        "Precision": precision_score(y_test_h, y_pred),
        "Recall": recall_score(y_test_h, y_pred),
        "F1": f1_score(y_test_h, y_pred),
        "ROC_AUC": roc_auc_score(y_test_h, y_prob)
    })

    # TP = Not Potable (0)
    cm = confusion_matrix(
        y_test_h, y_pred, labels=[1, 0]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Potable", "Not Potable"]
    )

    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(model_name)

plt.tight_layout()
plt.show()

holdout_df = pd.DataFrame(holdout_results)

print("===== HOLD-OUT RESULTS (70/30) =====")
print((holdout_df.set_index("Algorithm") * 100).round(2))


# In[15]:


metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

cv_scores = results_df.groupby("Algorithm")[metrics].mean() * 100
holdout_scores = holdout_df.set_index("Algorithm")[metrics] * 100

comparison_table = pd.concat(
    {
        "5-Fold CV": cv_scores,
        "Hold-Out 70/30": holdout_scores
    },
    axis=1
)

print("===== CV vs HOLD-OUT =====")
print(comparison_table.round(2).to_string())


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

# Average of 5-Fold CV
cv_avg = results_df.groupby("Algorithm")[metrics].mean() * 100

# Hold-Out results
holdout_avg = holdout_df.set_index("Algorithm")[metrics] * 100

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(19, 6))

for ax, model_name in zip(axes, models.keys()):

    cv_values = cv_avg.loc[model_name].values
    holdout_values = holdout_avg.loc[model_name].values

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(
        x - width/2, cv_values, width,
        label="5-Fold CV", color="steelblue"
    )

    bars2 = ax.bar(
        x + width/2, holdout_values, width,
        label="Hold-Out 70/30", color="orange"
    )

    ax.set_title(model_name, fontsize=14)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30)
    ax.set_ylim(0, 110)
    ax.legend()

    ax.bar_label(bars1, fmt="%.1f", padding=2)
    ax.bar_label(bars2, fmt="%.1f", padding=2)

plt.suptitle(
    "5-Fold Cross-Validation vs Hold-Out 70/30",
    fontsize=17
)

plt.tight_layout()
plt.show()


# ## drop missing value + hold out

# **Alternative treatment of missing values:** Delete every row with a missing value instead of filling it. This leaves 2,011 rows out of 3,276; the following results compare this approach with median imputation under Hold-Out evaluation.

# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import pandas as pd
import numpy as np

# Drop rows containing missing values
df_clean = df.dropna().copy()

print("Original Rows:", len(df))
print("Remaining Rows:", len(df_clean))
print("Deleted Rows:", len(df) - len(df_clean))

# Separate features and target
X_clean = df_clean.drop(columns=["Potability"])
y_clean = df_clean["Potability"]

# Hold-Out 70/30
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_clean, y_clean,
    test_size=0.30,
    stratify=y_clean,
    random_state=42
)

print("Training:", len(X_train_d))
print("Testing:", len(X_test_d))

# Scaling for Logistic Regression
scaler_d = StandardScaler()

X_train_scaled_d = scaler_d.fit_transform(X_train_d)
X_test_scaled_d = scaler_d.transform(X_test_d)


# In[18]:


drop_results = []

for model_name, model_template in models.items():

    model = clone(model_template)

    if model_name == "Logistic Regression":
        X_tr = X_train_scaled_d
        X_te = X_test_scaled_d
    else:
        X_tr = X_train_d
        X_te = X_test_d

    model.fit(X_tr, y_train_d)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    drop_results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy_score(y_test_d, y_pred),
        "Precision": precision_score(y_test_d, y_pred, zero_division=0),
        "Recall": recall_score(y_test_d, y_pred, zero_division=0),
        "F1": f1_score(y_test_d, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test_d, y_prob)
    })

drop_df = pd.DataFrame(drop_results)

print("===== DROP MISSING VALUES - HOLD-OUT 70/30 =====")
print((drop_df.set_index("Algorithm") * 100).round(2))


# In[19]:


import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

median_scores = holdout_df.set_index("Algorithm")[metrics] * 100
drop_scores = drop_df.set_index("Algorithm")[metrics] * 100

fig, axes = plt.subplots(1, 3, figsize=(19, 6))

for ax, model_name in zip(axes, models.keys()):

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(
        x - width/2,
        median_scores.loc[model_name],
        width,
        label="Median Imputation",
        color="steelblue"
    )

    bars2 = ax.bar(
        x + width/2,
        drop_scores.loc[model_name],
        width,
        label="Drop Missing",
        color="orange"
    )

    ax.set_title(model_name)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30)
    ax.set_ylim(0, 110)
    ax.legend()

    ax.bar_label(bars1, fmt="%.1f", padding=2)
    ax.bar_label(bars2, fmt="%.1f", padding=2)

plt.suptitle(
    "Median Imputation vs Drop Missing Values",
    fontsize=16
)

plt.tight_layout()
plt.show()


# ## drop missing value + 5 fold cross validation

# **5-Fold comparison:** Repeat cross-validation after dropping incomplete rows, then compare Accuracy, Precision, Recall, and F1 with the median-imputation approach.

# In[20]:


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)
import pandas as pd

# 1. Remove rows with missing values
df_clean = df.dropna().copy()

X = df_clean.drop(columns=["Potability"])
y = df_clean["Potability"]

print("Original Rows:", len(df))
print("Remaining Rows:", len(df_clean))
print("Deleted Rows:", len(df) - len(df_clean))

# 2. Stratified 5-Fold CV
skf = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

drop_cv_results = []

# 3. Train all 3 models
for model_name, model_template in models.items():

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y), start=1
    ):

        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Scaling for Logistic Regression
        if model_name == "Logistic Regression":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        model = clone(model_template)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        drop_cv_results.append({
            "Algorithm": model_name,
            "Fold": fold,
            "Train Accuracy": train_acc * 100,
            "Validation Accuracy": val_acc * 100,
            "Gap (pp)": (train_acc - val_acc) * 100,
            "Precision": precision_score(
                y_val, val_pred, zero_division=0
            ) * 100,
            "Recall": recall_score(
                y_val, val_pred, zero_division=0
            ) * 100,
            "F1": f1_score(
                y_val, val_pred, zero_division=0
            ) * 100
        })

# 4. Average across 5 folds
drop_cv_df = pd.DataFrame(drop_cv_results)

drop_cv_summary = (
    drop_cv_df.groupby("Algorithm")
    .mean(numeric_only=True)
    .drop(columns=["Fold"])
)

print("===== DROP MISSING + 5-FOLD CV =====")
print(drop_cv_summary.round(2).to_string())


# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Recall", "F1"]

# CV with Median Imputation
median_cv = (
    results_df.groupby("Algorithm")[metrics].mean() * 100
)

# CV after Dropping Missing Rows
drop_cv = drop_cv_df.groupby("Algorithm")[
    ["Validation Accuracy", "Precision", "Recall", "F1"]
].mean()

drop_cv = drop_cv.rename(
    columns={"Validation Accuracy": "Accuracy"}
)

# Comparison Table
comparison = pd.concat(
    {
        "Median Imputation": median_cv,
        "Drop Missing": drop_cv
    },
    axis=1
)

print("===== 5-FOLD CV COMPARISON =====")
print(comparison.round(2).to_string())

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, model_name in zip(axes, models.keys()):

    x = np.arange(len(metrics))
    width = 0.35

    b1 = ax.bar(
        x - width/2,
        median_cv.loc[model_name, metrics],
        width,
        label="Median Imputation",
        color="steelblue"
    )

    b2 = ax.bar(
        x + width/2,
        drop_cv.loc[model_name, metrics],
        width,
        label="Drop Missing",
        color="orange"
    )

    ax.set_title(model_name, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.legend()

    ax.bar_label(b1, fmt="%.1f", padding=2)
    ax.bar_label(b2, fmt="%.1f", padding=2)

plt.suptitle(
    "5-Fold CV: Median Imputation vs Drop Missing",
    fontsize=17
)

plt.tight_layout()
plt.show()


# ## check overfitting | underfitting 

# **Overfitting check:** Compare training accuracy with validation/testing accuracy. A large positive gap suggests that a model fits the training data much better than unseen data.

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.metrics import accuracy_score

results = []

# 4 Experiments
for missing_method in ["Median Imputation", "Drop Missing"]:

    if missing_method == "Drop Missing":
        data = df.dropna().copy()
    else:
        data = df.copy()

    X = data.drop(columns=["Potability"])
    y = data["Potability"]

    for evaluation in ["Hold-Out 70/30", "5-Fold CV"]:

        for model_name, model_template in models.items():

            steps = []

            # Median Imputation
            if missing_method == "Median Imputation":
                steps.append(
                    ("imputer", SimpleImputer(strategy="median"))
                )

            # Scaling for Logistic Regression
            if model_name == "Logistic Regression":
                steps.append(("scaler", StandardScaler()))

            steps.append(("model", clone(model_template)))

            pipeline = Pipeline(steps)

            # HOLD-OUT
            if evaluation == "Hold-Out 70/30":

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.30,
                    stratify=y,
                    random_state=42
                )

                pipeline.fit(X_train, y_train)

                train_acc = accuracy_score(
                    y_train, pipeline.predict(X_train)
                ) * 100

                test_acc = accuracy_score(
                    y_test, pipeline.predict(X_test)
                ) * 100

            # 5-FOLD CV
            else:

                cv = StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=42
                )

                scores = cross_validate(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    scoring="accuracy",
                    return_train_score=True
                )

                train_acc = scores["train_score"].mean() * 100
                test_acc = scores["test_score"].mean() * 100

            results.append({
                "Method": missing_method,
                "Evaluation": evaluation,
                "Model": model_name,
                "Train Accuracy": train_acc,
                "Test Accuracy": test_acc,
                "Gap (pp)": train_acc - test_acc
            })

# Results Table
overfit_df = pd.DataFrame(results)

print("===== OVERFITTING CHECK =====")
print(overfit_df.round(2).to_string(index=False))


# ============================================
# VISUALIZATION
# ============================================

experiments = [
    ("Median Imputation", "Hold-Out 70/30"),
    ("Median Imputation", "5-Fold CV"),
    ("Drop Missing", "Hold-Out 70/30"),
    ("Drop Missing", "5-Fold CV")
]

fig, axes = plt.subplots(2, 2, figsize=(17, 11))

for ax, (method, evaluation) in zip(axes.flat, experiments):

    subset = overfit_df[
        (overfit_df["Method"] == method) &
        (overfit_df["Evaluation"] == evaluation)
    ]

    x = np.arange(len(subset))
    width = 0.35

    b1 = ax.bar(
        x - width/2,
        subset["Train Accuracy"],
        width,
        label="Training",
        color="steelblue"
    )

    b2 = ax.bar(
        x + width/2,
        subset["Test Accuracy"],
        width,
        label="Testing / Validation",
        color="orange"
    )

    ax.set_title(f"{method} + {evaluation}", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(subset["Model"], rotation=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend()

    ax.bar_label(b1, fmt="%.1f", padding=2)
    ax.bar_label(b2, fmt="%.1f", padding=2)

plt.suptitle(
    "Overfitting / Underfitting Comparison - All Experiments",
    fontsize=17
)

plt.tight_layout()
plt.show()


# ## hyperparameter tuning | random forest 

# **Random Forest tuning:** Use `GridSearchCV` inside each training fold to select hyperparameters. The stored results show that the average train–validation gap decreases from **33.03** to **9.85 percentage points**, while validation accuracy changes from **66.97%** to **65.72%**. Tuning reduces overfitting here but does not improve validation accuracy.

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV
)
from sklearn.metrics import accuracy_score

X = df.drop(columns=["Potability"])
y = df["Potability"]

cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

param_grid = {
    "rf__max_depth": [5, 10],
    "rf__min_samples_split": [2, 10],
    "rf__min_samples_leaf": [2, 5]
}

results = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1
        ))
    ])

    # Before Tuning
    baseline.fit(X_train, y_train)

    # After Tuning - inside training fold only
    tuned_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1
        ))
    ])

    grid = GridSearchCV(
        tuned_pipeline,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    for name, model in [
        ("Before Tuning", baseline),
        ("After Tuning", grid.best_estimator_)
    ]:
        train_acc = accuracy_score(
            y_train, model.predict(X_train)
        ) * 100

        val_acc = accuracy_score(
            y_val, model.predict(X_val)
        ) * 100

        results.append({
            "Fold": fold,
            "Model": name,
            "Train Accuracy": train_acc,
            "Validation Accuracy": val_acc,
            "Gap": train_acc - val_acc
        })

    print(f"Fold {fold}/5 completed")

# Average results
result_df = pd.DataFrame(results)

summary = result_df.groupby("Model")[
    ["Train Accuracy", "Validation Accuracy", "Gap"]
].mean()

summary = summary.reindex([
    "Before Tuning", "After Tuning"
])

print("\n===== RANDOM FOREST 5-FOLD CV =====")
print(summary.round(2))

# Visualization
ax = summary[[
    "Train Accuracy", "Validation Accuracy"
]].plot(
    kind="bar",
    figsize=(9, 5),
    color=["steelblue", "orange"],
    rot=0
)

ax.set_title("Random Forest: Before vs After Tuning - 5-Fold CV")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 110)
ax.bar_label(ax.containers[0], fmt="%.1f")
ax.bar_label(ax.containers[1], fmt="%.1f")

plt.tight_layout()
plt.show()

