from flask import Flask, render_template,request, jsonify

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

import json

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

@app.route('/predict/',methods=['POST'])
def predict():

    if request.method=='POST':
        model, labels = load_model_and_labels()
        request_dictionary = request.form.to_dict()
        print(request_dictionary)
        values = list(request_dictionary.values())
    
        float_vals = [(float(x) if x else 0) for x in values]

        print(float_vals)
        new_vector = np.array(float_vals).reshape(1, -1)
        predicted_values = model.predict(new_vector)
        precicted = np.array2string(predicted_values).replace("[",  '').replace("]",  '')
        print(type(precicted))
        labels['predicted_diagnosis'] = precicted
        

        new_dict = {}
        for k in labels.keys():
            new_dict[str(k)] = labels[k]
        
        
        #print(new_dict)
        
        jsonStr = json.dumps(new_dict)
        print(jsonStr)
        return jsonStr
    #else:
    #    return "Hello"

	

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 9090))
	#run the app locally on the givn port
	app.run(host='127.0.0.1', port=port)
	#optional if we want to run in debugging mode
	app.run(debug=True)