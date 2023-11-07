from PIL import Image
import numpy as np
from utils import preprocess_data
from joblib import load


from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


# @app.route("/")
# def hello_world():
#     return render_template("template.html")


@app.route("/")
def index():
    return render_template("compare_image.html")


@app.route("/sum/<x>/<y>")
def sum_num(x, y):
    x = int(x)
    y = int(y)
    return "<p>sum is " + str(x + y) + "</p>"


@app.route("/model", methods=["POST"])
def pred_model():
    data = request.get_json()
    x = int(data["x"])
    y = int(data["y"])
    result = x + y
    return f"<p>sum is {result}</p>"


@app.route("/compare_images", methods=["POST"])
def compare_images():
    try:
        image1 = Image.open(request.files["image1"])
        image2 = Image.open(request.files["image2"])

        # Preprocess the images and convert them to numpy arrays
        image1 = np.array(image1.resize((8, 8)))
        image2 = np.array(image2.resize((8, 8)))

        # Flatten the images to make them compatible with the classifier
        image1 = image1.reshape(1, -1)
        image2 = image2.reshape(1, -1)

        model_path = "models/best_model_C-1_gamma-0.001.joblib"
        svm = load(model_path)
        digit1 = svm.predict(image1)
        digit2 = svm.predict(image2)

        # Compare the predicted digits
        are_same = digit1[0] == digit2[0]

        return jsonify({"are_same": are_same})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("server is running")
    app.run()
