# import numpy as np
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.linear_model import LogisticRegression

import pickle

from flask import Flask
from flask import request
from flask import jsonify

app = Flask('Credit card')
### Importing the python pickle model

dv = 'dv.bin'
model = 'model2.bin'

with open(dv, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(model, 'rb') as f_in:
    model = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0,1]
    result = {
        "credit_card_probability": float(y_pred)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
