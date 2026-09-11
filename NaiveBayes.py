import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# -----------------------------------------------------------
# Metadatos del problema (para mostrar en las páginas HTML)
# -----------------------------------------------------------
DATA_SOURCE = "Online Shoppers Purchasing Intention Dataset (UCI / Kaggle)"
INDEPENDENT_VAR_NAME = "Valor de la página (PageValues)"
INDEPENDENT_VAR_UNIT = "puntos"
TARGET_VAR_NAME = "Revenue (Compra)"
CLASS_0_MEANING = "No compró"
CLASS_1_MEANING = "Sí compró"

# -----------------------------------------------------------
# 1. Cargar datos
# -----------------------------------------------------------
df = pd.read_csv("data/online_shoppers_intention.csv")
df = df[["PageValues", "Revenue"]].dropna()
df["Revenue"] = df["Revenue"].astype(int)

NUM_RECORDS = len(df)

# -----------------------------------------------------------
# 2. Preparar X, y y hacer el split 80/20
# -----------------------------------------------------------
x = df[["PageValues"]]
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 3. Entrenar el modelo (se entrena una sola vez al iniciar la app)
# -----------------------------------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -----------------------------------------------------------
# 4. Calcular métricas sobre el 20% de prueba
# -----------------------------------------------------------
_y_pred_test = model.predict(X_test)

CONFUSION_MATRIX = confusion_matrix(y_test, _y_pred_test)
ACCURACY = round(accuracy_score(y_test, _y_pred_test), 4)
PRECISION = round(precision_score(y_test, _y_pred_test, zero_division=0), 4)
RECALL = round(recall_score(y_test, _y_pred_test), 4)
F1 = round(f1_score(y_test, _y_pred_test), 4)


# -----------------------------------------------------------
# 5. Función para el formulario (usada desde app.py)
# -----------------------------------------------------------
def predecir_compra(page_values):
    """Recibe el PageValues y devuelve la clase + probabilidad."""
    pred = model.predict([[page_values]])[0]
    prob = model.predict_proba([[page_values]])[0][1]

    return {
        "clase": int(pred),
        "mensaje": CLASS_1_MEANING if pred == 1 else CLASS_0_MEANING,
        "probabilidad": round(float(prob) * 100, 2),  # en porcentaje
    }