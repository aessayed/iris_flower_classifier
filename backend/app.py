from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from model import load_model

app = Flask(__name__)
CORS(app)  # Allow frontend to call this backend

model = load_model()

@app.route('/')
def home():
    return 'Iris Prediction API is up and running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        if not features or len(features) != 4:
            return jsonify({"error": "You must send 4 features"}), 400

        features = torch.FloatTensor([features])
        with torch.no_grad():
            output = model(features)
            predicted_class = torch.argmax(output).item()

        class_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

        return jsonify({
            "prediction": class_map[predicted_class],
            "class_id": predicted_class
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
