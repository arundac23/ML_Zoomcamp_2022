#import numpy as np
#
import bentoml
from bentoml.io import JSON

from pydantic import BaseModel

class HeartfailureApplication(BaseModel):
    age: int
    sex: str
    chest_pain_type: str
    resting_bp_s: int
    cholesterol: int
    fasting_blood_sugar: str
    resting_ecg: str
    max_heart_rate: int
    exercise_angina: str
    oldpeak: float
    st_slope: str


model_ref = bentoml.sklearn.get("heart_failure_prediction:ox7di6s7bsnaajna")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("heart_failure_classifier", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=HeartfailureApplication), output=JSON())
async def classify(heartfailure_application):
    application_data = heartfailure_application.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)
    result = prediction[0]

    if result > 0.5:
        return {
            "status": "POSSIBLE CHANCE OF HEART FAILURE"
        }
    elif result > 0.25:
        return {
            "status": "SOME POSSIBLITIY OF HEART FAILURE"
        }
    else:
        return {
            "status": "NO POSSIBLITIY OF HEART FAILURE"
        }