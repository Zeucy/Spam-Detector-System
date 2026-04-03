import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def main():
    print("Loading data...")
    # Load dataset. 'latin-1' encoding is typically needed for this specific spam dataset.
    data_path = 'spam.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path, encoding='latin-1')

    print("Cleaning data...")
    # Keep only the relevant columns and rename them for clarity
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # Map labels to binary values: 0 for ham (not spam), 1 for spam
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Drop any potential rows with missing values
    df.dropna(inplace=True)

    print("Splitting data into training and test sets...")
    X = df['message']
    y = df['label']
    
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Vectorizing text data...")
    # Convert text strings to numerical features using TF-IDF
    # We remove english stop words as they usually don't carry spam/ham signals
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Naive Bayes model...")
    # Train the Multinomial Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    print("Evaluating model...")
    # Predict on the test set
    y_pred = model.predict(X_test_vec)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))

    print("Saving the model and vectorizer...")
    # Save the trained model and vectorizer to disk
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Saved as 'spam_model.pkl' and 'tfidf_vectorizer.pkl'. Training complete.")

if __name__ == "__main__":
    main()
