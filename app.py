import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS



# --- 1. Create the Flask App ---
# '__name__' tells Flask where to look for files (like the 'templates' folder)
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests



# --- 2. Load the Saved Model ---
model_filename = 'best_titanic_model.joblib'
print(f"--- Loading model from {model_filename} ---")

try:
    model = joblib.load(model_filename)
    print("--- Model loaded successfully ---")

except FileNotFoundError:
    print(f"---!!! ERROR: Model file '{model_filename}' not found. ---")
    print("---!!! Please run 'titanic_final_project.py' first to create it. ---")
    model = None

except Exception as e:
    print(f"Error loading model: {e}")
    model = None





# --- 3. Define App Routes ---


# Route 1: The Homepage (Serves your index.html)
@app.route('/')
def home():
    """
    This route serves the main 'index.html' page.
    Flask automatically looks in the 'templates' folder for this file.
    """
    return render_template('index.html')



# Route 2: The Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives data from the HTML form and returns a prediction.
    This is the endpoint that your index.html's JavaScript will call.
    """

    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded.'})

    try:
        # Get the JSON data sent from the frontend
        data = request.get_json()

        # Convert the received data into a pandas DataFrame
        # This must match the features your model was trained on
        features = [
            data['pclass'],
            data['sex'],
            data['age'],
            data['sibsp'],
            data['parch'],
            data['fare'],
            data['embarked']
        ]

        df = pd.DataFrame([features], columns=[
            'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'
        ])

        # --- 4. Make the Prediction ---
        # The loaded pipeline handles all preprocessing (imputing, scaling, etc.)
        prediction = model.predict(df)
        result_text = "Survived" if prediction[0] == 1 else "Did Not Survive"

        # Return the result as JSON
        return jsonify({
            'success': True,
            'prediction_text': result_text,
            'prediction_value': int(prediction[0])
        })

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })



# --- 5. Run the App ---
if __name__ == '__main__':
    # 'port=5000' is the default for Flask
    # 'host='0.0.0.0'' makes it accessible on your local network
    # 'debug=True' gives helpful error messages
    app.run(host='0.0.0.0', port=5000, debug=True)
