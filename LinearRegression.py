import pandas as pd
from sklearn.linear_model import LinearRegression

# scikit-learn
# .\.venv\Scripts\python.exe -m pip install scikit-learn

# --- Dataset info (Point 3: Dataset) ---
DATASET_PATH = "data/crop_yield_dataset.csv"
INDEPENDENT_VAR_NAME = "Fertilizer Amount"
DEPENDENT_VAR_NAME = "Crop Yield"
INDEPENDENT_VAR_UNIT = "kg/hectare"
DEPENDENT_VAR_UNIT = "tons/hectare"
DATA_SOURCE = (
    "Synthetic dataset generated for academic purposes, simulating a realistic "
    "relationship between fertilizer application and crop yield based on typical "
    "agronomic ranges."
)

# Load the dataset
dataframe = pd.read_csv(DATASET_PATH)
NUM_RECORDS = len(dataframe)

# Independent variable (X)
x = dataframe[["fertilizer_kg_ha"]]

# Dependent variable (y)
y = dataframe[["yield_ton_ha"]]

# Train the model
model = LinearRegression()
model.fit(x, y)


def calculate_yield(fertilizer_amount):
    """Predicts crop yield given a fertilizer amount, using the trained model."""
    result = model.predict([[fertilizer_amount]])[0][0]
    return result