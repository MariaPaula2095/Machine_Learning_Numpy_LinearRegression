import pandas as pd
import matplotlib.pyplot as plt
import base64
from sklearn.linear_model import LinearRegression
#scikit-learn
#.\.venv\Scripts\python.exe -m pip install scikit-learn

#archivo profesor en teams
data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

# variable   se le pasa para que procese
datafrme = pd.DataFrame(data)

#pdie varibale dependediente e independiente

#variable independiente
x = datafrme[["Study Hours"]]

#variable dependiente
y = datafrme[["Final Grade"]]

#llamar regresion
model = LinearRegression()

#funcion entrenamiento
model.fit(x,y)

#utilizar modelo 
#definir funcion y exponerla
def calculateGrade(hours):
    result = model.predict([[hours]])[0] #cuando no pasamos mas variables de retorno se pone 0
    return result