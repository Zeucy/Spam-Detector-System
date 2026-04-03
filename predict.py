import joblib
import sys
import os

def load_objects():
    model_path = 'spam_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Model or vectorizer not found.")
        print("Please run 'python train.py' first to train the model.")
        return None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_message(message, model, vectorizer):
    # Vectorize the incoming message using the loaded TF-IDF vectorizer
    vec_message = vectorizer.transform([message])
    
    # Predict using the loaded model
    prediction = model.predict(vec_message)[0]
    
    # Calculate probability if needed
    probabilities = model.predict_proba(vec_message)[0]
    spam_prob = probabilities[1]
    
    label = "Spam" if prediction == 1 else "Ham (Not Spam)"
    return label, spam_prob

def interactive_mode(model, vectorizer):
    print("\n--- SMS Spam Detector ---")
    print("Type a message to check if it's spam or ham.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            message = input("Enter message: ")
            if message.lower() in ['exit', 'quit']:
                break
            if not message.strip():
                continue
                
            label, spam_prob = predict_message(message, model, vectorizer)
            
            print(f"\nResult: {label}")
            print(f"Spam Probability: {spam_prob:.2%}\n")
            
        except KeyboardInterrupt:
            break
            
    print("\nExiting spam detector.")

if __name__ == "__main__":
    model, vectorizer = load_objects()
    if model and vectorizer:
        # Check if a message was passed as a command-line argument
        if len(sys.argv) > 1:
            message = " ".join(sys.argv[1:])
            label, spam_prob = predict_message(message, model, vectorizer)
            print(f"Message: {message}")
            print(f"Result: {label} (Spam Probability: {spam_prob:.2%})")
        else:
            # If no argument is passed, go into interactive mode
            interactive_mode(model, vectorizer)
