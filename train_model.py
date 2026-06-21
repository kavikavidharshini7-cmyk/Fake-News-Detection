import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("news.csv")

X = df["text"].astype(str)
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,3),
    max_features=10000
)

X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y,
    test_size=0.2,
    random_state=42
)

# Model
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# SAVE model + vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Model saved successfully!")