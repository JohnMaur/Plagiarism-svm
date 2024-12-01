from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset for training 
documents = [
    "This is an original text document.",
    "This document is an original piece of content.",
    "This is plagiarized content taken from another source.",
    "Plagiarized content often mirrors other works exactly."
]
labels = [0, 0, 1, 1]  # 0 = original, 1 = plagiarized

# Train SVM Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Flask App Setup
app = Flask(__name__)

@app.route("/check-plagiarism", methods=["POST"])
def check_plagiarism():
    try:
        data = request.json
        content_to_check = data.get("text", "")
        
        if not content_to_check.strip():
            return jsonify({"error": "No content provided"}), 400
        
        # Transform input text to TF-IDF vector
        input_vector = vectorizer.transform([content_to_check]).toarray()
        
        # Predict plagiarism probability
        prediction = svm_model.predict(input_vector)[0]
        probability = svm_model.predict_proba(input_vector)[0][1] * 100  # Plagiarism confidence
        
        response = {
            "isPlagiarized": bool(prediction),
            "confidence": probability
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001)
