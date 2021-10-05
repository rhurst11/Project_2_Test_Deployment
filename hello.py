import numpy as np
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/model")
def model():
    return render_template("model.html")


#To use the predict button in our web-app
@app.route("/predict", methods=["POST"])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)   
    
    return render_template("predict.html", prediction_text = 'Classifier result for end user is : {}'.format(output))



if __name__ == "__main__":
    
    app.run(debug = True)