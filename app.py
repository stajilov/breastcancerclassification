from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
#import keras.models
#for regular expressions, saves time dealing with string data
import re

#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
#initialize these variables
#model, graph = init()
from sklearn.externals import joblib


model_file_name = 'breast_prediction.pkl'
labels_file_name = 'labels.pkl'



def load_model_and_labels():
    model = joblib.load(model_file_name) 
    labels = joblib.load(labels_file_name)
    return model, labels


@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
    model, labels = load_model_and_labels()
    response = 1

    

    email = request.form.get('email')
    name = request.form.get('name')
    print(email, name)
    return email	
	

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	app.run(debug=True)