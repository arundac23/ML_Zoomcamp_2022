### BentoMl Production
BentoML is an open platform that simplifies ML model deployment and enables you to serve your models at production scale in a short duration.
XGboost model from lesson 6 is saved as bento model using following command
```python
# Save xgboost model
bentoml.xgboost.save_model('credit_risk_model',
                            model,
                            custom_objects={'DictVectorizer': dv})
```
BentoML will generator a directory in the home directory where the model will be kept.

Once the model is saved, we can create a `service.py` file that will be used to define the BentoML service:

```python
import bentoml
from bentoml.io import JSON


# Pull the model as model reference (it pulls all the associate metadata of the model)
model_ref = bentoml.xgboost.get('credit_risk_model:latest')
# Call DictVectorizer object using model reference
dv = model_ref.custom_objects['DictVectorizer']
# Create the model runner (it can also scale the model separately)
model_runner = model_ref.to_runner()

# Create the service 'credit_risk_classifier' and pass the model
svc = bentoml.Service('credit_risk_classifier', runners=[model_runner])


# Define an endpoint on the BentoML service
@svc.api(input=JSON(), output=JSON()) # decorate endpoint as in json format for input and output
def classify(application_data):
    # transform data from client using dictvectorizer
    vector = dv.transform(application_data)
    # make predictions using 'runner.predict.run(input)' instead of 'model.predict'
    prediction = model_runner.predict.run(vector)
    
    result = prediction[0] # extract prediction from 1D array
    print('Prediction:', result)

    if result > 0.5:
        return {'Status': 'DECLINED'}
    elif result > 0.3:
        return {'Status': 'MAYBE'}
    else:
        return {'Status': 'APPROVED'}
```

Once the service and the endpoint is created we can run the app using the command: `bentoml serve service:svc`, where ***service*** is the script and ***svc*** is the service name.
- When running bento service instead of using `http://0.0.0.0:3000/`, use `http://localhost:3000/

##  Deploying Your Prediction Service
In this section we are going to look at BentoML cli and what operations BentoML is performing behind the scenes.

We can get a list of saved model in the terminal using the commmand `bentoml models list`. This command shows all the saved models and their tags, module, size, and the time they were created at.

We can use `bentoml models list -o json|yaml|table` to display the output in one of the given format.

Running the command `bentoml models get credit_risk_model:modelname` displays the information about the model which looks like:

```yaml
name: credit_risk_model
version: model name
module: bentoml.xgboost
labels: {}
options:
  model_class: Booster
metadata: {}
context:
  framework_name: xgboost
  framework_versions:
    xgboost: 1.6.2
  bentoml_version: 1.0.7
  python_version: 3.10.6
signatures:
  predict:
    batchable: false
api_version: v2
creation_time: '2022-10-20T08:29:54.706593+00:00'
```

Important thing to note here is that the version of the XGBoost in the `framework_versions` has to be same as the model was trained with otherwise we might get inconsistent results. The BentoML pulls these dependencies automatically and generates this file for convenience.

The next we want to do is, creating the file `bentofile.yaml`:

```yaml
service: "service.py:svc" # Specify entrypoint and service name
labels: # Labels related to the project for reminder (the provided labels are just for example)
  owner: bentoml-team
  project: gallery
include:
- "*.py" # A pattern for matching which files to include in the bento build
python:
  packages: # Additional pip packages required by the service
    - xgboost
    - sklearn
```

Once we have our `service.py` and `bentofile.yaml` files ready we can build the bento by running the command `bentoml build`. It will look in the service.py file to get all models being used and into bentofile.yaml file to get all the dependencies and creates one single deployable directory for us. The output will look something like this:

```bash
Successfully built Bento(tag="credit_risk_classifier:model name")
```

We can look into this directory by locating `cd ~/bentoml/bentos/credit_risk_classifier/modelname/` and the file structure may look like this:

```bash
.
├── README.md # readme file
├── apis
│   └── openapi.yaml # openapi file to enable Swagger UI
├── bento.yaml # bento file to bind everything together
├── env # environment related directory
│   ├── docker # auto generate dockerfile (also can be customized)
│   │   ├── Dockerfile
│   │   └── entrypoint.sh
│   └── python # requirments for installation
│       ├── install.sh
│       ├── requirements.txt
│       └── version.txt
├── models # trained model(s)
│   └── credit_risk_model
│       ├── modelname
│       │   ├── custom_objects.pkl # custom objects (in our case DictVectorizer)
│       │   ├── model.yaml # model metadate
│       │   └── saved_model.ubj # saved model
│       └── latest
└── src
    └── service.py # bentoml service file for endpoint
```

The idea behind the structure like this is to provide standardized way that a machine learning service might required.

Now the last thing we need to do is to build the docker image. This can be done with `bentoml containerize credit_risk_classifier:model_name`.

> Note: We need to have Docker installed before running this command.

Once the docker image is built successfully, we can run `docker run -it --rm -p 3000:3000 containerize credit_risk_classifier:model_name` to see if everything is working as expected. We are exposing 3000 port to map with the service port which is also 3000 and this should take us to Swagger UI page again.

