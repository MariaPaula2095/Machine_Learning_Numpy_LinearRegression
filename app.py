import io
import base64
import math        
import pandas as pd
import matplotlib 
matplotlib.use('Agg') # Required for Flask to render plots in the background
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

# Import your custom modules
import LinearRegression
import NaiveBayes
import LogisticRegressionModel as logistic_model 

app = Flask(__name__)

# --- HELPER FUNCTION FOR CONFUSION MATRIX PLOT ---
def generate_confusion_matrix_plot(matrix, title):
    """
    Generates a heatmap image (Base64) for a given confusion matrix.
    """
    import numpy as np
    
    cm = np.array(matrix)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    classes = ['No Purchase (0)', 'Purchase (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90, va="center")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight='bold')
            
    plt.ylabel('Actual Outcome')
    plt.xlabel('Predicted Outcome')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()
    
    return plot_url

# --- ROUTES R1A1 ---
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
    
    plt.scatter(LinearRegression.x, LinearRegression.y, color='blue', alpha=0.5, label='Actual Data')
    plt.plot(LinearRegression.x, LinearRegression.model.predict(LinearRegression.x), color='black', linewidth=2, label='Regression Line')

    plt.title("Linear Regression: Fertilizer vs Crop Yield")
    plt.xlabel(f"{LinearRegression.INDEPENDENT_VAR_NAME} ({LinearRegression.INDEPENDENT_VAR_UNIT})")
    plt.ylabel(f"{LinearRegression.DEPENDENT_VAR_NAME} ({LinearRegression.DEPENDENT_VAR_UNIT})")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if fertilizer_input is not None and prediction_result is not None:
        plt.scatter([fertilizer_input], [prediction_result], color='red', s=100, zorder=5, label=f'Prediction ({fertilizer_input})')
    
    plt.legend()

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

# --- ROUTES R1A2: LOGISTIC REGRESSION ---
@app.route('/supervised/logistic/concepts')
def logistic_concepts():
    return render_template('logistic_concepts.html')

@app.route('/supervised/logistic/application', methods=['GET', 'POST'])
def logistic_application():
    prediction_result = None
    error_message = None
    page_values_input = None # Cambiado de time_input a page_values_input
    plot_url = None

    if request.method == "POST":
        raw_value = request.form.get("page_values_input", "").strip() # Captura el nuevo input del HTML

        if raw_value == "":
            error_message = "Please enter a PageValues amount."
        else:
            try:
                page_values_input = float(raw_value)
                prediction_result = logistic_model.predict_purchase(page_values_input)
            except ValueError:
                error_message = "The value must be numeric."

    # --- Generate the plot ---
    plt.figure(figsize=(8, 5))

    colors = logistic_model.y.map({0: "red", 1: "green"})
    plt.scatter(
        logistic_model.x["PageValues"], # Apunta a PageValues
        logistic_model.y,
        c=colors,
        alpha=0.3,
        label="Sessions (red=No Purchase, green=Purchase)"
    )

    if page_values_input is not None and prediction_result is not None:
        plt.scatter(
            [page_values_input], [prediction_result["predicted_class"]],
            color="blue", s=150, zorder=5,
            label=f"Prediction ({page_values_input} points)"
        )

    plt.title("Logistic Regression: PageValues vs Purchase")
    plt.xlabel(f"{logistic_model.INDEPENDENT_VAR_NAME} ({logistic_model.INDEPENDENT_VAR_UNIT})")
    plt.ylabel("Purchase (0 = No, 1 = Yes)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()
    # --- End Plot ---

    return render_template(
        "logistic_application.html",
        result=prediction_result,
        error=error_message,
        page_values_input=page_values_input, # Enviado al HTML
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
    plot_url = generate_confusion_matrix_plot(
        logistic_model.CONFUSION_MATRIX, 
        "Logistic Regression: Confusion Matrix"
    )
    return render_template(
        'logistic_metrics.html',
        accuracy=logistic_model.ACCURACY,
        precision=logistic_model.PRECISION,
        recall=logistic_model.RECALL,
        f1=logistic_model.F1,
        plot_url=plot_url
    )
    
# --- ROUTES R1A2: NAIVE BAYES ---
@app.route("/naive-bayes/concepts")
def naive_bayes_concepts():
    return render_template("naive_bayes_concepts.html")

@app.route("/naive-bayes/application", methods=["GET", "POST"])
def naive_bayes_application():
    prediction_result = None
    error_message = None
    page_values_input = None
    plot_url = None

    if request.method == "POST":
        raw_value = request.form.get("page_values_input", "").strip()

        if raw_value == "":
            error_message = "Please enter a PageValues amount."
        else:
            try:
                page_values_input = float(raw_value)
                prediction_result = NaiveBayes.predict_purchase(page_values_input)
            except ValueError:
                error_message = "The value must be numeric."

    # --- Generate the plot ---
    plt.figure(figsize=(8, 5))

    colors = NaiveBayes.y.map({0: "red", 1: "green"})
    plt.scatter(
        NaiveBayes.x["PageValues"],
        NaiveBayes.y,
        c=colors,
        alpha=0.3,
        label="Sessions (red=No Purchase, green=Purchase)"
    )

    if page_values_input is not None and prediction_result is not None:
        plt.scatter(
            [page_values_input], [prediction_result["predicted_class"]],
            color="blue", s=150, zorder=5,
            label=f"Prediction ({page_values_input} points)"
        )

    plt.title("Naive Bayes: PageValues vs Purchase")
    plt.xlabel(f"{NaiveBayes.INDEPENDENT_VAR_NAME} ({NaiveBayes.INDEPENDENT_VAR_UNIT})")
    plt.ylabel("Purchase (0 = No, 1 = Yes)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()
    # --- End Plot ---

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
        plot_url=plot_url
    )
    
@app.route("/naive-bayes/metrics")
def naive_bayes_metrics():
    plot_url = generate_confusion_matrix_plot(
        NaiveBayes.CONFUSION_MATRIX, 
        "Naive Bayes: Confusion Matrix"
    )
    return render_template(
        "naive_bayes_metrics.html",
        accuracy=NaiveBayes.ACCURACY,
        precision=NaiveBayes.PRECISION,
        recall=NaiveBayes.RECALL,
        f1=NaiveBayes.F1,
        variable_independiente=NaiveBayes.INDEPENDENT_VAR_NAME,
        variable_objetivo=NaiveBayes.TARGET_VAR_NAME,
        plot_url=plot_url
    )


# --- ROUTES R2A2: K-MEANS CONCEPTS & APPLICATION 

@app.route('/unsupervised/kmeans')
def kmeans_simulation_route():
    return render_template('kmeans_view.html') 

@app.route('/unsupervised/kmeans/concepts')
def kmeans_concepts():
    return render_template('kmeans_concepts.html')

@app.route('/unsupervised/kmeans/application', methods=['GET', 'POST'])
def kmeans_application():
    prediction_result = None
    error_message = None
    dance_input = None
    energy_input = None
    plot_url = None

    # Final centroids based on your kmeans_simulation.py output for Iteration 3
    # Cluster 1: [0.2849, 0.2542] (red)
    # Cluster 2: [0.7368, 0.4479] (blue)
    # Cluster 3: [0.6524, 0.8213] (green)
    centroids = {
        "Cluster 1 (Low Energy, Low Danceability)": {"x": 0.2849, "y": 0.2542, "color": "red"},
        "Cluster 2 (High Danceability, Moderate Energy)": {"x": 0.7368, "y": 0.4479, "color": "blue"},
        "Cluster 3 (High Energy, High Danceability)": {"x": 0.6524, "y": 0.8213, "color": "green"}
    }

    if request.method == "POST":
        raw_dance = request.form.get("dance_input", "").strip()
        raw_energy = request.form.get("energy_input", "").strip()

        if raw_dance == "" or raw_energy == "":
            error_message = "Please enter both Danceability and Energy values."
        else:
            try:
                dance_input = float(raw_dance)
                energy_input = float(raw_energy)

                if not (0.0 <= dance_input <= 1.0) or not (0.0 <= energy_input <= 1.0):
                    error_message = "Values must be between 0.0 and 1.0."
                else:
                    # Find nearest centroid using Euclidean distance
                    min_dist = float('inf')
                    best_cluster = None
                    best_color = None

                    for name, data in centroids.items():
                        # Calculate Euclidean distance
                        dist = math.dist((dance_input, energy_input), (data["x"], data["y"]))
                        if dist < min_dist:
                            min_dist = dist
                            best_cluster = name
                            best_color = data["color"]

                    prediction_result = {
                        "cluster_name": best_cluster,
                        "color": best_color
                    }
            except ValueError:
                error_message = "Values must be numeric."

    # --- Generate the plot ---
    plt.figure(figsize=(8, 5))

    # Plot original dataset points in the background
    try:
        # Usando la ruta generada por tu simulador
        df = pd.read_csv("data/spotify_100_tracks.csv")
        plt.scatter(df['Danceability'], df['Energy'], color='gray', alpha=0.2, label="Dataset Tracks")
    except Exception as e:
        print(f"Error reading CSV: {e}") # Ayuda para depurar si no encuentra el archivo
        pass

    # Plot final centroids
    for name, data in centroids.items():
        plt.scatter([data["x"]], [data["y"]], color=data["color"], marker='X', s=200, edgecolor='black', label=f"Centroid: {name.split(' ')[1]}")

    # Plot user input if prediction exists
    if prediction_result and dance_input is not None and energy_input is not None:
        plt.scatter(
            [dance_input], [energy_input], 
            color="orange", marker='*', s=400, edgecolor='black', zorder=10, 
            label="Your Track"
        )

    plt.title("K-Means: Track Profiling")
    plt.xlabel("Danceability (0.0 - 1.0)")
    plt.ylabel("Energy (0.0 - 1.0)")
    plt.xlim(0, 1) # Asegurar límites del gráfico consistentes con la simulación
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Place legend outside plot to avoid hiding data
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()
    # --- End Plot ---

    return render_template(
        'kmeans_application.html',
        result=prediction_result,
        error=error_message,
        dance_input=dance_input,
        energy_input=energy_input,
        plot_url=plot_url
    )
if __name__ == "__main__":
    app.run(debug=True)