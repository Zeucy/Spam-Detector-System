from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Load the model and vectorizer at startup
model_path = 'spam_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully.")
else:
    print("Error: Models not found! Make sure to run train.py first.")
    model, vectorizer = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json()
    message = data.get('message', '')

    if not message.strip():
        return jsonify({"error": "Message cannot be empty."}), 400

    # Vectorize and predict
    vec_message = vectorizer.transform([message])
    prediction = model.predict(vec_message)[0]
    probabilities = model.predict_proba(vec_message)[0]
    spam_prob = float(probabilities[1])

    is_spam = bool(prediction == 1)

    return jsonify({
        "is_spam": is_spam,
        "probability": spam_prob,
        "message": message
    })

if __name__ == '__main__':
    # Run the app locally over port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
