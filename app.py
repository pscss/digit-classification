from PIL import Image
import numpy as np
from utils import preprocess_data
import joblib
import json


from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("template.html")


# @app.route("/")
# def index():
#     return render_template("predict.html")


# @app.route("/")
# def index():
#     return render_template("compare_image.html")


@app.route("/sum/<x>/<y>")
def sum_num(x, y):
    x = int(x)
    y = int(y)
    return "<p>sum is " + str(x + y) + "</p>"


# @app.route("/model", methods=["POST"])
# def pred_model():
#     data = request.get_json()
#     x = int(data["x"])
#     y = int(data["y"])
#     result = x + y
#     return f"<p>sum is {result}</p>"


# def _read_image(img):
#     img = img.resize((8, 8))  # Resize to 8x8 pixels
#     img = img.convert("L")  # Convert to grayscale

#     # Convert the image to a numpy array and flatten it
#     img_array = np.array(img)
#     image_array = (img_array / 16).astype(int)
#     for i in range(8):
#         for j in range(8):
#             if image_array[i][j] > 0:
#                 image_array[i][j] = 16 - image_array[i][j]
#     img_flattened = preprocess_data(np.array([image_array]))
#     return img_flattened


# @app.route("/compare_images", methods=["POST"])
# def compare_images():
#     try:
#         image1 = Image.open(request.files["image1"])
#         image2 = Image.open(request.files["image2"])

#         # Preprocess the images and convert them to numpy arrays
#         image1 = _read_image(image1)
#         image2 = _read_image(image2)
#         # image1 = np.array(image1.resize((8, 8)))
#         # image2 = np.array(image2.resize((8, 8)))

#         # # Flatten the images to make them compatible with the classifier
#         # image1 = image1.reshape(1, -1)
#         # image2 = image2.reshape(1, -1)

#         model_path = "models/best_model_C-1_gamma-0.001.joblib"
#         svm = load(model_path)
#         digit1 = svm.predict(image1)
#         digit2 = svm.predict(image2)

#         # Compare the predicted digits
#         are_same = digit1[0] == digit2[0]

#         return f"<p>Images are same: {are_same}.</p>"
#     except Exception as e:
#         return jsonify({"error": str(e)})


# Define a route to handle image upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file:
        image = _read_image(Image.open(file))
        model_path = "models/best_model_C-1_gamma-0.001.joblib"
        model = joblib.load(model_path)
        prediction = model.predict(image)
        return jsonify({"prediction": str(prediction[0])})
    else:
        return jsonify({"error": "Invalid file format"})


@app.route("/prediction", methods=["POST"])
def prediction():
    data_json = request.json
    if data_json:
        data_dict = json.loads(data_json)
        image = np.array([data_dict["image"]])
        model_path = "models/best_model_C-1_gamma-0.001.joblib"
        model = joblib.load(model_path)
        try:
            prediction = model.predict(image)
            return jsonify({"prediction": str(prediction[0])})
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "Invalid data format"})


if __name__ == "__main__":
    print("server is running")
    app.run(host="0.0.0.0", port=8000)
