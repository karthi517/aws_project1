import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)[0]
    if prediction==0:
        prediction='Not Having Heart Disease'
    else:
        prediction='Having Heart Disease'

    return render_template("index.html", prediction_text = "The Patient is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)