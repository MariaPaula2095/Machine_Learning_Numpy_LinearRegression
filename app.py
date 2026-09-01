import io
import base64
import matplotlib 
matplotlib.use('Agg') # is important to flask 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import LinearRegression

app = Flask(__name__)


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
        plot_url=plot_url  # <--- ESTO ES LO QUE FALTA
    )


if __name__ == "__main__":
    app.run(debug=True)