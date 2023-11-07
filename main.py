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
    js = request.get_json()
    x = js["x"]
    y = js["y"]
    x = int(x)
    y = int(y)
    return "<p>sum is " + str(x + y) + "</p>"


if __name__ == "__main__":
    app.run(debug=True)
