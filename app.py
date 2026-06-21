from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html", prediction=None, news_text="")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    # vectorize input
    data = vectorizer.transform([news])

    # prediction
    result = model.predict(data)[0]

    # SAFE mapping
    if str(result).upper() == "REAL" or result == 1:
        prediction = "Real News"
    else:
        prediction = "Fake News"

    return render_template(
        "index.html",
        prediction=prediction,
        news_text=news
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)