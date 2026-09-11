import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# -----------------------------------------------------------
# Problem Metadata (to be displayed on HTML pages)
# -----------------------------------------------------------
DATA_SOURCE = "Online Shoppers Purchasing Intention Dataset (UCI / Kaggle)"
INDEPENDENT_VAR_NAME = "Page Values"
INDEPENDENT_VAR_UNIT = "Points"
TARGET_VAR_NAME = "Revenue (Purchase)"
CLASS_0_MEANING = "No Purchase"
CLASS_1_MEANING = "Purchase"

# -----------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------
df = pd.read_csv("data/online_shoppers_intention.csv")

# Extract only the necessary columns and drop null values
df = df[["PageValues", "Revenue"]].dropna()

# Ensure the target variable is integer (0 or 1)
df["Revenue"] = df["Revenue"].astype(int)

NUM_RECORDS = len(df)

# -----------------------------------------------------------
# 2. Prepare X, y and perform 80/20 split
# -----------------------------------------------------------
# Keep 'x' and 'y' global so app.py can use them for plotting
x = df[["PageValues"]]
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 3. Train the model (trained only once when the app starts)
# -----------------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------------------------------------
# 4. Calculate real metrics on the 20% test set
# -----------------------------------------------------------
_y_pred_test = model.predict(X_test)

# Convert to list so the confusion matrix plot doesn't fail
CONFUSION_MATRIX = confusion_matrix(y_test, _y_pred_test).tolist()
ACCURACY = round(accuracy_score(y_test, _y_pred_test), 4)
PRECISION = round(precision_score(y_test, _y_pred_test, zero_division=0), 4)
RECALL = round(recall_score(y_test, _y_pred_test), 4)
F1 = round(f1_score(y_test, _y_pred_test), 4)

# -----------------------------------------------------------
# 5. Form function (used by app.py)
# -----------------------------------------------------------
def predict_purchase(page_values):
    """
    Receives PageValues points and returns the predicted class, message, and probability.
    """
    pred = model.predict([[page_values]])[0]
    prob = model.predict_proba([[page_values]])[0][1]

    return {
        "predicted_class": int(pred),
        "message": CLASS_1_MEANING if pred == 1 else CLASS_0_MEANING,
        "probability": round(float(prob) * 100, 2),  # Converted to percentage
    }