import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(dataset_path="data/chat_dataset.csv",
                model_path="models/chat_model.pkl",
                vectorizer_path="models/tfidf_vectorizer.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    df = pd.read_csv(dataset_path)
    X = df['message']
    y = df['label']

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Model and vectorizer trained and saved")
    return model, vectorizer

def main():
    train_model()

if __name__ == "__main__":
    main()
