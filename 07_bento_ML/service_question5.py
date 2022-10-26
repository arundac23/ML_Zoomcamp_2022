import bentoml
from bentoml.io import JSON, NumpyNdarray 
import numpy as np

## model from curl -O https://s3.us-west-2.amazonaws.com/bentoml.com/mlzoomcamp/coolmodel.bentomodel
model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")


model_runner = model_ref.to_runner()

svc = bentoml.Service("HW07_prediction", runners=[model_runner])

@svc.api(input=NumpyNdarray(shape=(1, 4), dtype=np.float32, enforce_shape=True), output=JSON())
async def classify(vector):
    prediction = model_runner.predict.run(vector)

    return prediction
