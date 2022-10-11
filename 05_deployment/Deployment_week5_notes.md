# Deploying Machine Learning Models
## Overview

Fifth week of Machine learning zoomcamp is about learning the model deployment.In overall, Trained Model need to be saved from the Jupyter notebook into python bin file. Then load it into a web service, for which we will use [Flask API Python library](https://flask.palletsprojects.com/en/2.0.x/). Also, we will use pipenv virtual environment to create a Python environment to manage software dependencies and [Docker](https://www.docker.com/products/docker-desktop) to create a container for handling system dependencies. Finally, we will deploy the container in the cloud with AWS EB.

Using the _Churn_ exercises from weeks 3 and 4, a production environment for our Churn model is created:

![overall_process](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/05-deployment/images/thumbnail-5-01.jpg)

## Saving and loading a model

We used [Pickle library](https://docs.python.org/3/library/pickle.html) to save and load a machine learning model. In general, pickle allows us to save Python model.bin file. All the necessary libraries need to loaded before importing the model again.
* `model.bin` --> The model that extracted from jupyter notebook.
* `open(input_file, 'rb')` - open a binary file with the name assigned as input_file , which has read permissiono only , which can be for writing ('w')  Writing permission is used for creating files and reading is applied for loading files. A binary file contains bits instead of text.
* `x.close()` - close the x file. It is important to guarantee that the file contains the object saved with pickle.
* `with open(input_file, 'rb'):` - same as `open(input_file, 'rb')`, but in this case you guarantee that this file will be closed.
* `pickle.load(x)` - Pickle function to load a python object x.
* `pickle.dump((dv, model), f_out)` -Pickle dump function to save a python object file.

## Web services: Introduction to Flask

A web service is a software system that supports interoperable machine-to-machine interaction over a network. In general, users make requests with some data and get the required prediction response from server.

app = Flask('churn') --> Create a flask app for the churn prediction

Flask has different decorators to handle http requests
Different methods for retrieving data from a specified URL are defined in this protocol. The following table summarizes the different http methods:

Request 	          Purpose
 * `GET`	  --> The most common method. A GET message is send, and the server returns data
 * `POST`	  --> Used to send HTML form data to the server. The data received by the POST method is not cached by the server.
 * `HEAD`	 -->  Same as GET method, but no response body.
 * `PUT`	 -->  Replace all current representations of the target resource with uploaded content.
 * `DELETE`	--> Deletes all current representations of the target resource given by the URL.

In the churn prediction deployment, We used 
@app.route('/predict', methods=['POST'])
The method associated with this web service was POST because it was required to send some information about customers, which is not easy to do with GET method. 
All the requests and responses  in flask must be in JSON files, which are quite similar to Python dictionaries.
The `gunicorn` library helps us to prepare a model to be launched in production. Note: This gunicorn will work on WSL environment. Waitress need to be used for windows machine.

## Creating python virtual environments

`pipenv` is used to create virtual environments on which model is developed,tested and deployed that application.This is helpful to manage software level dependencies. It allows to install specific versions of software libraries in order to reproduce the consistent results.

## Deploy the model with containers

Container is next level of isolation from virtual environment level separation. It is helpful to manage the system level dependencies. Docker container is used for deploying this churn model. All the necessary softwares, webservices, virtual environment can be installed inside the docker container without affecting the external system.

## Push the container to the cloud

Finally clould services like Amazon Web Services Elastic Beanstalk or GCP are used to deploy it to a production environment by pushing from a docker container 
