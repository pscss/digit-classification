# from flask import Flask

# app = Flask(__name__)


# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"


from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("template.html")


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


if __name__ == "__main__":
    print("server is running")
    app.run(host="0.0.0.0", port=5000)
