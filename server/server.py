import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import cross_origin, CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# app = Flask(__name__)
app = Flask(__name__, static_folder="static")
CORS(app)

# Load the saved model
model_path = os.path.join("model", "my_model2.h5")
loaded_model = tf.keras.models.load_model(model_path)

# Load the input text vectorizer
vectorizer_path = os.path.join("model", "vectorizer.pkl")
with open(vectorizer_path, "rb") as f:
    vectorizer_config = pickle.load(f)
    vectorizer_weights = pickle.load(f)

loaded_vectorizer = tf.keras.layers.TextVectorization.from_config(vectorizer_config)
loaded_vectorizer.set_weights(vectorizer_weights)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict2", methods=["POST", "GET"])
@cross_origin()
def predict():
    data = request.get_json()
    print("data: ", data)

    input_text = data["input"]

    # Preprocess the input text using the loaded vectorizer
    vectorized_text = loaded_vectorizer(input_text)
    vectorized_text = np.expand_dims(vectorized_text, 0)

    # Make prediction using the loaded model
    prediction = loaded_model.predict(vectorized_text)

    # Apply threshold (0.5) to convert probabilities to True/False
    threshold = 0.5
    binary_prediction = (prediction > threshold).astype(int)
    classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    int_predictions = binary_prediction[0].tolist()
    print(binary_prediction)
    fl = []
    for i in range(5):
        if int_predictions[i] == 1:
            fl.append(classes[i])

    response = {"predicted_class": fl}

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
