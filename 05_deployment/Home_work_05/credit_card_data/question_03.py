# import numpy as np

# from sklearn.feature_extraction import DictVectorizer
# from sklearn.linear_model import LogisticRegression

import pickle

dv = 'dv.bin'
model = 'model1.bin'

with open(dv, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(model, 'rb') as f_in:
    model = pickle.load(f_in)

client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

X = dv.transform([client])

y_pred = model.predict_proba(X)[0,1]
print (f'For the input client information {client}')
print (f'The probability of getting a credit card for this client is {round(y_pred,3)}')