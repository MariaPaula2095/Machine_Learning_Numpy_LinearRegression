
#importa clase flask
from flask import Flask, render_template

# Crea una instancia de esa clase y la guarda en la variable app
# Es una variable especial que Python asigna automáticamente a cada módulo 
# (archivo .py). Su valor depende de cómo se ejecuta el archivo:
# Si ejecutas el archivo directamente (python app.py), __name__ vale "__main__".
# Si el archivo se importa desde otro módulo, __name__ vale el nombre del archivo (ej. "app").
app = Flask (__name__)

@app.route("/")
def home():
    return "Hello World"

@app.route("/template")
def template():
    return render_template("index.html")