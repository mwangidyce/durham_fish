from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


def predict(arr: list):
    with open("fish_clf.model", "rb") as f:
        clf = pickle.load(f)
    return list(clf.predict(np.array(arr).reshape(1, -1)))[0]


@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/classify", methods=["POST"])
def classify_api():
    user_input = request.get_json()
    input_arr = [int(each) for each in user_input.values()]
    prediction = predict(input_arr)
    print(prediction)

    return {"prediction": prediction}
