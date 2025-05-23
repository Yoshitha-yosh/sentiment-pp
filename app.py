from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

# Load separate models
sentiment_model = load_model("sentiment_model.h5")
platform_model = load_model("platform_model.h5")

# Load tokenizer and encoders
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
sentiment_encoder = pickle.load(open("sentiment_encoder.pkl", "rb"))
platform_encoder = pickle.load(open("platform_encoder.pkl", "rb"))

MAX_LEN = 100  # same maxlen as training

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "")
        if not text.strip():
            return jsonify({"error": "Empty text"}), 400

        # Preprocess input text
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)

        # Predict sentiment
        sentiment_pred = sentiment_model.predict(padded)
        sentiment_label = sentiment_encoder.inverse_transform([np.argmax(sentiment_pred)])[0]

        # Predict platform
        platform_pred = platform_model.predict(padded)
        platform_label = platform_encoder.inverse_transform([np.argmax(platform_pred)])[0]

        return jsonify({
            "sentiment": sentiment_label,
            "platform": platform_label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
