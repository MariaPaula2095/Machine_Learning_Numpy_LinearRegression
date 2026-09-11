import io
import base64
import matplotlib 
matplotlib.use('Agg') # is important to flask 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import LinearRegression
import logistic_model
import NaiveBayes


app = Flask(__name__)

#ROUTES R1A1
@app.route("/")
def home():
    return render_template("hello_world.html")


@app.route("/template")
def template():
    return render_template("index.html")


@app.route("/types")
def types():
    return render_template("types.html")


@app.route("/LinearRegression", methods=["GET", "POST"])
def calculate():
    prediction_result = None
    error_message = None
    fertilizer_input = None

    if request.method == "POST":
        raw_value = request.form.get("fertilizer", "").strip()

        if raw_value == "":
            error_message = "Please enter a fertilizer amount."
        else:
            try:
                fertilizer_input = float(raw_value)
                prediction_result = LinearRegression.calculate_yield(fertilizer_input)
            except ValueError:
                error_message = "The value must be numeric."


# --- START PLOT GENERATION ---
    plt.figure(figsize=(8, 5))
    
    # 1. Scatter plot for actual data points
    plt.scatter(LinearRegression.x, LinearRegression.y, color='blue', alpha=0.5, label='Actual Data')
    
    # 2. Plot the regression line
    plt.plot(LinearRegression.x, LinearRegression.model.predict(LinearRegression.x), color='black', linewidth=2, label='Regression Line')

    # Label configuration
    plt.title("Linear Regression: Fertilizer vs Crop Yield")
    plt.xlabel(f"{LinearRegression.INDEPENDENT_VAR_NAME} ({LinearRegression.INDEPENDENT_VAR_UNIT})")
    plt.ylabel(f"{LinearRegression.DEPENDENT_VAR_NAME} ({LinearRegression.DEPENDENT_VAR_UNIT})")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Add the prediction point if the user submitted the form
    if fertilizer_input is not None and prediction_result is not None:
        plt.scatter([fertilizer_input], [prediction_result], color='red', s=100, zorder=5, label=f'Prediction ({fertilizer_input})')
    
    plt.legend()

    # Save the plot to memory (Base64)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close() 
    # --- END PLOT GENERATION ---
    

    return render_template(
        "linearRegression.html",
        result=prediction_result,
        error=error_message,
        fertilizer_input=fertilizer_input,
        num_records=LinearRegression.NUM_RECORDS,
        independent_var_name=LinearRegression.INDEPENDENT_VAR_NAME,
        dependent_var_name=LinearRegression.DEPENDENT_VAR_NAME,
        independent_var_unit=LinearRegression.INDEPENDENT_VAR_UNIT,
        dependent_var_unit=LinearRegression.DEPENDENT_VAR_UNIT,
        data_source=LinearRegression.DATA_SOURCE,
        plot_url=plot_url  
    )

#routes R1A2

@app.route('/supervised/logistic/concepts')
def logistic_concepts():
    return render_template('logistic_concepts.html')


@app.route('/supervised/logistic/application', methods=['GET', 'POST'])
def logistic_application():
    resultado = None
    error_message = None
    tiempo_input = None

    if request.method == "POST":
        raw_value = request.form.get("tiempo", "").strip()

        if raw_value == "":
            error_message = "Por favor ingresa un valor de tiempo."
        else:
            try:
                tiempo_input = float(raw_value)
                if tiempo_input < 0:
                    error_message = "El tiempo no puede ser negativo."
                else:
                    resultado = logistic_model.predecir_compra(tiempo_input)
            except ValueError:
                error_message = "El valor debe ser numérico."

    # --- Generar la gráfica (igual patrón que Linear Regression) ---
    plt.figure(figsize=(8, 5))

    colores = logistic_model.y.map({0: "red", 1: "green"})
    plt.scatter(
        logistic_model.x["ProductRelated_Duration"],
        logistic_model.y,
        c=colores,
        alpha=0.3,
        label="Sesiones (rojo=No compró, verde=Sí compró)"
    )

    if tiempo_input is not None and resultado is not None:
        plt.scatter(
            [tiempo_input], [resultado["clase"]],
            color="blue", s=150, zorder=5,
            label=f"Predicción ({tiempo_input}s)"
        )

    plt.title("Regresión Logística: Tiempo en el sitio vs Compra")
    plt.xlabel(f"{logistic_model.INDEPENDENT_VAR_NAME} ({logistic_model.INDEPENDENT_VAR_UNIT})")
    plt.ylabel("Compra (0 = No, 1 = Sí)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()
    # --- Fin gráfica ---

    return render_template(
        "logistic_application.html",
        result=resultado,
        error=error_message,
        tiempo_input=tiempo_input,
        num_records=logistic_model.NUM_RECORDS,
        independent_var_name=logistic_model.INDEPENDENT_VAR_NAME,
        target_var_name=logistic_model.TARGET_VAR_NAME,
        class_0=logistic_model.CLASS_0_MEANING,
        class_1=logistic_model.CLASS_1_MEANING,
        data_source=logistic_model.DATA_SOURCE,
        plot_url=plot_url,
    )


@app.route('/supervised/logistic/metrics')
def logistic_metrics():
    return render_template(
        'logistic_metrics.html',
        matriz=logistic_model.CONFUSION_MATRIX,
        accuracy=logistic_model.ACCURACY,
        precision=logistic_model.PRECISION,
        recall=logistic_model.RECALL,
        f1=logistic_model.F1,
    )
    
@app.route("/naive-bayes/concepts")
def naive_bayes_concepts():
    return render_template("naive_bayes_concepts.html")


@app.route("/naive-bayes/application", methods=["GET", "POST"])
def naive_bayes_application():
    prediction_result = None
    error_message = None
    page_values_input = None

    if request.method == "POST":
        raw_value = request.form.get("page_values", "").strip()

        if raw_value == "":
            error_message = "Please enter a PageValues amount."
        else:
            try:
                page_values_input = float(raw_value)
                prediction_result = NaiveBayes.predecir_compra(page_values_input)
            except ValueError:
                error_message = "The value must be numeric."

    return render_template(
        "naive_bayes_application.html",
        result=prediction_result,
        error=error_message,
        page_values_input=page_values_input,
        num_records=NaiveBayes.NUM_RECORDS,
        independent_var_name=NaiveBayes.INDEPENDENT_VAR_NAME,
        independent_var_unit=NaiveBayes.INDEPENDENT_VAR_UNIT,
        target_var_name=NaiveBayes.TARGET_VAR_NAME,
        class_0=NaiveBayes.CLASS_0_MEANING,
        class_1=NaiveBayes.CLASS_1_MEANING,
        data_source=NaiveBayes.DATA_SOURCE,
    )
    
@app.route("/naive-bayes/metrics")
def naive_bayes_metrix():
    return render_template(
        "naive_bayes_metrics.html",
        matriz=NaiveBayes.CONFUSION_MATRIX,
        accuracy=NaiveBayes.ACCURACY,
        precision=NaiveBayes.PRECISION,
        recall=NaiveBayes.RECALL,
        f1=NaiveBayes.F1,
        variable_independiente=NaiveBayes.INDEPENDENT_VAR_NAME,
        variable_objetivo=NaiveBayes.TARGET_VAR_NAME,
    )


if __name__ == "__main__":
    app.run(debug=True)
