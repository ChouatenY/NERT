from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

# Load the trained SVM model
try:
    model = joblib.load('svm_ner_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define a mapping of NER classes to colors
label_color_map = {
    0: 'green',   # Example: Person
    1: 'blue',    # Example: Organization
    2: 'red',     # Example: Location
    3: 'orange',  # Example: Miscellaneous
    4: 'purple'   # Example: Other
}

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded properly'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        words = text.split()
        predictions = []

        for word in words:
            # For SVM, we might need a feature vector, but this is a placeholder
            word_length = len(word)
            prediction = model.predict(np.array([[word_length]]))
            color = label_color_map.get(int(prediction[0]), 'grey')  # Default to grey if label not found
            predictions.append({'word': word, 'color': color})

        return jsonify(predictions)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
