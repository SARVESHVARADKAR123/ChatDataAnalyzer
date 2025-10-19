import pandas as pd
import pickle
from datetime import datetime
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class ChatPredictor:
    def __init__(self, model_path="models/chat_model.pkl", vectorizer_path="models/tfidf_vectorizer.pkl"):
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Model/vectorizer not found. Train first.")

    def predict_messages(self, messages: List[str], threshold: float = 0.5) -> List[Dict]:
        X = self.vectorizer.transform(messages)
        probs = self.model.predict_proba(X)[:,1]
        results = []
        for msg, prob in zip(messages, probs):
            results.append({
                "message": msg,
                "suspicious_probability": prob,
                "is_suspicious": prob > threshold
            })
        return results

    def add_and_retrain(self, message: str, sender: str = "Manual", label: int = 1,
                        dataset_path: str = "data/chat_dataset.csv",
                        model_path: str = "models/chat_model.pkl",
                        vectorizer_path: str = "models/tfidf_vectorizer.pkl"):
        df = pd.read_csv(dataset_path)
        new_entry = pd.DataFrame({
            "timestamp":[pd.Timestamp.now()],
            "sender":[sender],
            "message":[message],
            "label":[label]
        })
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(dataset_path, index=False)
        
        print(f"[INFO] Message flagged: '{message}' by sender '{sender}'")
        print("[INFO] Retraining model...")

        # Retrain model
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500)
        X_vec = self.vectorizer.fit_transform(df['message'])
        self.model = LogisticRegression(max_iter=1000)
        y = df['label']
        self.model.fit(X_vec, y)

        # Save updated model/vectorizer
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print("[INFO] Model retrained and saved successfully ✅")
        return True
