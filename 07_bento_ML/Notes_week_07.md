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
## Sending, Receiving and Validating Data:

Data validation is another great feature on BentoML that ensures the data transferation is valid and reliable. We can integrate Python library Pydatic with BentoML for this purpose.

Pydantic can be installed with `pip install pydantic`, after that we need to import the `BaseModel` class from the library and create our custom class for data validation:

```python
# Create pydantic base class to create data schema for validation
class CreditApplication(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int
```

Our model is trained on 13 features of different data types and the BaseModel will ensure that we are always recieving them for the model prediction.

Next we need to implement pass the class in our bentoml service:

```python
# Pass pydantic class in the application
@svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON()) # decorate endpoint as in json format for input and output
def classify(credit_application):
    # transform pydantic class to dict to extract key-value pairs 
    application = credit_application.dict()
    # transform data from client using dictvectorizer
    vector = dv.transform(application)
    # make predictions using 'runner.predict.run(input)' instead of 'model.predict'
    prediction = model_runner.predict.run(vector) 
```

Along the `JSON()`, BentoML uses various other descriptors in the input and output specification of the service api, for example, NumpyNdarray(), PandasDataFrame(), Text(), and many more.

## High-Performance Serving
BentoML can optimize the performance on our application where the model will have to make predictions on hundreds of requests per seconds. For this we need to install locust (`pip install locust`), which is a Python open-source library for load testing.

Once the locust is installed, we'll need to create `locustfile.py` and implement user flows for testing:

```python
import numpy as np
from locust import task
from locust import between
from locust import HttpUser


# Sample data to send
sample = {"seniority": 3,
 "home": "owner",
 "time": 36,
 "age": 26,
 "marital": "single",
 "records": "no",
 "job": "freelance",
 "expenses": 35,
 "income": 0.0,
 "assets": 60000.0,
 "debt": 3000.0,
 "amount": 800,
 "price": 1000
 }

# Inherit HttpUser object from locust
class CreditRiskTestUser(HttpUser):
    """
    Usage:
        Start locust load testing client with:
            locust -H http://localhost:3000, in case if all requests failed then load client with:
            locust -H http://localhost:3000 -f locustfile.py

        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    # create mathod with task decorator to send request
    @task
    def classify(self):
        self.client.post("/classify", json=sample) # post request in json format with the endpoint 'classify'

    wait_time = between(0.01, 2) # set random wait time between 0.01-2 secs
```

This first optimization we can implement in our application is called *async* optimization. This will make the application to process the requests in parallel and the model will make predictions simultaneously:

```python
# Define an endpoint on the BentoML service
# pass pydantic class application
@svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON()) # decorate endpoint as in json format for input and output
async def classify(credit_application): # parallelized requests at endpoint level (async)
    # transform pydantic class to dict to extract key-value pairs 
    application = credit_application.dict()
    # transform data from client using dictvectorizer
    vector = dv.transform(application)
    # make predictions using 'runner.predict.run(input)' instead of 'model.predict'
    prediction = await model_runner.predict.async_run(vector) # bentoml inference level parallelization (async_run)
```

Another optimization is to take advantage of micro-batching. This is another BentoML feature where it can combine the data coming from multiple users and combine them into **one array**, and then this array will be batched into smaller batches when the model prediction is called. There are few steps we need to do to enable this functionality, the first thing we have to save the model with bentoml `signatures` feature:

```python
# Save the model batchable settings for production efficiency
bentoml.xgboost.save_model('credit_risk_model',
                            model,
                            custom_objects={'DictVectorizer': dv},
                           signatures={  # model signatures for runner inference
                               'predict': { 
                                   'batchable': True, 
                                   'batch_dim': 0 # '0' means bentoml will concatenate request arrays by first dimension
                               }
                           })
```

Running `bentoml serve --production` will make the batchable model in serving, the `--production` flag will enable more than one process for our web workers.

We can also configure the batching parameters of the runner by creating `bentoconfiguration.yaml` file:

```python
# Config file controls the attributes of the runner
runners:
  batching:
    enabled: true
    max_batch_size: 100
    max_latency_ms: 500
```

> Note: In general, we are not supposed to be running the traffic generator on same machine that is serving the application requests because that takes away the CPU from the requests server.

