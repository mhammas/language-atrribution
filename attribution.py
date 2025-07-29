import csv
import json
import os
import argparse
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
import joblib

def load_csv_data(csv_path, text_column="description", class_column="class"):
    class_texts = defaultdict(list)
    all_texts = []
    all_labels = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row.get(text_column, "").strip()
            label = row.get(class_column, "").strip()
            if text and label.isdigit():
                label = int(label)
                class_texts[label].append(text)
                all_texts.append(text)
                all_labels.append(label)
    return class_texts, all_texts, all_labels

def load_json_comments(json_path):
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return [entry.get("comment", "").strip() for entry in data]

def get_top_keywords_per_class(class_texts, top_k=20):
    keywords = set()
    for label, texts in class_texts.items():
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf = vectorizer.fit_transform(texts)
        means = np.asarray(tfidf.mean(axis=0)).flatten()
        top_indices = means.argsort()[::-1][:top_k]
        class_keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        print(f"Class {label}: {class_keywords}")
        keywords.update(class_keywords)
    return sorted(keywords)

def vectorize_with_vocab(texts, vocab):
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    return vectorizer.fit_transform(texts)

def run_cross_validation(X, y, model, folds=10):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\n=== {folds}-Fold Cross Validation ===")
    print(f"Accuracy scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} | Std: {scores.std():.4f}")

def train_full_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\nModel saved to {path}")

def classify_comments(model, comment_vectors, comments):
    predictions = model.predict(comment_vectors)
    return [{"comment": c, "predicted_class": int(cls)} for c, cls in zip(comments, predictions)]

def main():
    parser = argparse.ArgumentParser(description="Comment Classifier with TF-IDF Keywords per Class")
    parser.add_argument("--csv", default="data.csv", help="Path to CSV file")
    parser.add_argument("--json", default="comments.json", help="Path to comments.json")
    parser.add_argument("--model-out", default="models/comment_model.pkl", help="Where to save trained model")
    parser.add_argument("--predictions-out", default="predictions.json", help="Where to save predictions")
    args = parser.parse_args()

    # Load and prepare text data
    class_texts, all_texts, all_labels = load_csv_data(args.csv)
    print(f"Loaded {len(all_texts)} descriptions across {len(class_texts)} classes.")

    # Get top keywords
    selected_keywords = get_top_keywords_per_class(class_texts, top_k=20)
    print(f"\nTotal unique keywords selected: {len(selected_keywords)}")

    # Vectorize using selected keywords
    X = vectorize_with_vocab(all_texts, selected_keywords)
    run_cross_validation(X, all_labels, RandomForestClassifier(), folds=10)

    model = train_full_model(X, all_labels)
    save_model(model, args.model_out)

    # Classify comments
    comments = load_json_comments(args.json)
    comment_vectors = vectorize_with_vocab(comments, selected_keywords)
    predictions = classify_comments(model, comment_vectors, comments)

    with open(args.predictions_out, "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to {args.predictions_out}")

if __name__ == "__main__":
    main()
