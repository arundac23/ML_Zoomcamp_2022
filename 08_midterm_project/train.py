#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import ExtraTreesClassifier


# In[2]:


data = 'heart.csv'
df = pd.read_csv(data)


# In[3]:


sex_values = {
    1: 'M',
    0: 'F',
}
df.sex = df.sex.map(sex_values)

fasting_blood_sugar_values = {
    0: '0',
    1: '1',
}
df.fasting_blood_sugar = df.fasting_blood_sugar.map(fasting_blood_sugar_values)

resting_ecg_values = {
    0: 'Normal',
    1: 'ST',
    2: 'LVH'
}
df.resting_ecg = df.resting_ecg.map(resting_ecg_values)

exercise_angina_values = {
    0: 'No',
    1: 'Yes'
}
df.exercise_angina = df.exercise_angina.map(exercise_angina_values)
ST_slope_values = {
    1: 'Up',
    2: 'Flat',
    3: 'Down'
}
df.ST_slope = df.ST_slope.map(ST_slope_values)


# In[4]:


df.columns = df.columns.str.lower()
categorical = list(df.dtypes[df.dtypes == 'object'].index)


# In[5]:


numerical = ['age', 'resting_bp_s', 'cholesterol','max_heart_rate','oldpeak']


# In[6]:


df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.heart_diesease.values
y_test = df_test.heart_diesease.values

del df_train['heart_diesease']
del df_test['heart_diesease']


# In[7]:


df_train = df_train.fillna('Down')
df_test = df_test.fillna('Down')


# In[8]:


dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)


test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)


# In[9]:


et = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=1)
model = et.fit(X_train, y_train)


# In[10]:


import bentoml


# In[11]:


bentoml.sklearn


# In[12]:


bentoml.sklearn.save_model(
    'heart_failure_prediction',
    model,
    custom_objects={
        'dictVectorizer': dv
    })


# In[13]:


import json


# In[14]:


request = df_test.iloc[0].to_dict()
print(json.dumps(request, indent=2))


# In[ ]:




