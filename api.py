import base64
from flask import Flask, request, render_template
from penguin_classify import classification

# instantiate flask app
app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    jBody = request.json
    image_data = base64.b64decode(jBody["body"])
    return classification(image_data)

@app.errorhandler(500)
def internal_error(exception):
    return str(exception), 500


if __name__ == "__main__":
    app.run(debug=False)
