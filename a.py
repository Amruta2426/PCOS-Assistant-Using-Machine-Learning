from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np


app = Flask(__name__, template_folder="template")

reg = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def hello_worl():
     return render_template("main.html")

@app.route("/about")
def hello_wor():
     return render_template("about.html")

@app.route("/choose")
def hello_wo():
     return render_template("choose.html")

@app.route("/remedy")
def hello_wr():
     return render_template("sol.html")

@app.route("/test")
def hello1():
     return render_template("test.html")


@app.route("/predict", methods=["POST"])
def home():
        
    float_features = [float(X) for X in request.form.values()]
    features =  [np.array(float_features)]
    
    prediction = reg.predict(features)
    print(prediction)
    return render_template("index.html", data=prediction)

reg1 = pickle.load(open("model1.pkl", "rb"))


@app.route("/test1")
def hello2():
     return render_template("test2.html")


@app.route("/predict", methods=["POST"])
def Profhome():
    float_features1 = [float(x) for x in request.form.values()]
    features1 =  [np.array(float_features1)]
    
    prediction1 = reg1.predict(features1)
    print(prediction1)
    return render_template("index.html", data=prediction1)


if __name__ == "__main__":
    app.run(debug=True)
