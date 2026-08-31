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
    )


if __name__ == "__main__":
    app.run(debug=True)