# AI Spam Detector Web App

A lightweight, machine-learning-powered web application that detects whether a given message is standard text ("Ham") or unsolicited/malicious junk ("Spam").

## 🚀 Features

- **Machine Learning**: Uses Python's `scikit-learn` (Multinomial Naive Bayes & TF-IDF Vectorizer).
- **Fast Web Backend**: A lightweight `Flask` API backend.
- **Premium Frontend**: A beautifully engineered "glassmorphism" aesthetic built with Vanilla JS, dark mode HTML/CSS, and zero bulky frontend frameworks.

## 🌐 Live Demo

You can try out the live version of this project hosted on Vercel here:
**[👉 Click here to view the Live Website](https://spam-detector-system.vercel.app)**


---

## 🛠️ How to Download and Run the Project

If you are downloading this project to your local machine, here is how you can get it running:

### 1. Install Dependencies
Open your command terminal inside this folder and run:
```bash
pip install -r requirements.txt
```

### 2. Start the Server
Start the local Flask app:
```bash
python app.py
```
Then open your web browser and navigate to `http://127.0.0.1:5000/`.

---

## 🧠 Changing the Dataset & Retraining the AI

If you want the AI to learn from a **new** dataset (e.g., detecting spam emails instead of SMS, or using a different language):

1. Find or create a new CSV dataset.
2. Replace `spam.csv`. (Ensure your new CSV has the exact same structure, or modify `train.py` to match the new column names!).
3. Run the training script:
   ```bash
   python train.py
   ```
   *This script will parse your new dataset, calculate the new TF-IDF patterns, and automatically overwrite `spam_model.pkl` and `tfidf_vectorizer.pkl` with the new "brain".*
4. Restart your Flask server (`python app.py`). The website will immediately start utilizing your brand-new dataset!

---

## 🎨 Customizing the Frontend UI

You can completely rebrand or restructure the website without touching the AI logic:

- **Text & Structure (`templates/index.html`)**: Open this file to change the site title, headers, or add new structural elements.
- **Colors & Styles (`static/style.css`)**: Look at the top `:root` section to easily swap out the global hex colors. You can also edit the `.orb` classes here to change the background gradients.
- **Animations & Logic (`static/script.js`)**: Modify this file if you want to change the delay times, the text of the result responses, or how the confident percentage bar behaves.
