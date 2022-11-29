### Intro to Serverless Deep Learning
Serverless is a computing service that runs code in response to events and automatically manages the computing resources required by that code. In this section, we are using AWS lambda.
### AWS lambda
AWS lambda is an event-driven, serverless computing platform provided by Amazon as a part of Amazon Web Services. The main advantage of lambda is that you're only charged when your code is requested.You won't be charged during the dowm time.
For creating a new lambda function,
* Search for lambda in AWS, select lambda services.
* Create a new function --> Enter function name --> runtime as python 3.9 --> Architecture = default selection.
* After entering the required information, code source will show your function and tests.

# TensorFlow Lite
***TensorFlow Lite*** (TFL) is a version of TensorFlow used for **inference** _only_. You cannot train new models with it, but you can predict results. This is useful for deployment as TFL is much smaller than the full version of the framework, thus making it possible to do inference on environments such as mobile or cloud, where storage and computing resources are either limited or expensive.
## Convert a Keras model to TF-Lite
![image](https://user-images.githubusercontent.com/76126029/204427631-b8891c42-f9c7-433a-a93c-87c307bbf804.png)
We need to use interpreter when using a TF-Lite model which slightly involved than a simple Keras model.
![image](https://user-images.githubusercontent.com/76126029/204428156-90309bcb-f7ad-44c9-82e9-73f32cb6677d.png)

Keras takes care of managing the inputs and outputs of the model for us, but TF-Lite does not.

`interpreter.get_input_details()` will give us all the details about the model's input; `interpreter.get_output_details()` will do the same for the outputs. We're interested in accessing the ***indices*** of both the input and the output. In our example, the output of both functions returns a dictionary list with a single dictionary each because our model only has one input and one output, so we can access the `index` key on each and store it:
![image](https://user-images.githubusercontent.com/76126029/204429116-ae139f50-369a-40ae-afae-53106f490580.png)
With our stored indices, we can now set the input, do inference and get our predictions:
![image](https://user-images.githubusercontent.com/76126029/204430424-9fb0ebaa-2205-40b3-9784-6e5ab5e45f04.png)
## Removing TF dependencies (image processing)

In the code above we're still loading the full TensorFlow library and we still need to access the image preprocessing functions that live in the TF libraries in order to preprocess our input.

Regarding image preprocessing, so far we've seen code similar to this:

```python
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input

img = load_img('image.jpg', target_size=(299,299))

x = np.array(img)
X = np.array([x])

X = preprocess_input(X)
```

We can replace `load_img()` with a similar function from `PIL` and we can reimplement `preprocess_input()` easily, thus getting rid of TF dependencies:

```python
from PIL import Image

with Image.open('image.jpg') as img:
    img = img.resize((299, 299), Image.NEAREST)

def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x

x = np.array(img, dtype='float32')
X = np.array([x])

X = preprocess_input(X)
```

* In `img.resize()`, `NEAREST` is the interpolation method we use for resizing, which also happens to be the default.
* By looking at Keras' source code, we're able to see that `preprocess_input()` does a very simple transformation to our input which we can implement in a simple function in our code.
* When converting the image to an array, we convert it to `float32` in order to avoid errors.

Alternatively, there's a library, `keras-image-helper`,  which contains preprocessors for a few convnets. Install it with `!pip install keras-image-helper`.

```python
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size(299, 299))
X = preprocessor.from_path('image.jpg')
```

## Removing TF dependencies (runtime)

In order to finally remove all TF dependencies, we must install `tflite_runtime`.

```python
!pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

We can now import it as a drop-in replacement for TF:

```python
import tflite_runtime.interpreter as tflite
```

And use all the code from the previous sections.

# Exporting code for Lambda

*Aside: you can convert Jupyter Notebooks to Python scripts with a built-in command line utility: `jupyter nbconvert --to script 'notebook.ipynb'`*

Once you've got a TFLite model working, you can create a Lambda script with a `lambda_handler()` method that calls a `predict()` function that contains your code.

We cannot import our model object directly to Lambda, but we can upload a Docker image instead that contains our `lambda_function.py` and our dependencies.

# Preparing a Docker image

We can use a publicly available Lambda Python base image from Amazon as our base image and add our code on it.

***WARNING:*** _Amazon Linux images are CentOS based; using code that has been tested on Debian-based distros may not work due to different available libraries and dependencies._

```dockerfile
FROM public.ecr.aws/lambda/python:3.8

RUN pip3 install keras_image_helper
RUN pip3 install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

COPY model.tflite .
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]
```

* `public.ecr.aws` is a URL that contains all publicly available Docker images from Amazon.
* We do not use `pipenv` in this Dockerfile because it's difficult to make it work with TensorFlow.
* TensorFlow's available package on `pip` is made for Debian-based distros and will not work on our image. We need to recompile TensorFlow for our image. Luckily, there are available precompiled images on [Alexey Grigorev's Repo](https://github.com/alexeygrigorev/tflite-aws-lambda).
    * `Wheel` is a built-package format supported by `pip`. Instead of downloading a package from pip's repo, we directly install a wheel from a URL.
* We don't need an entrypoint in this Docker image because it's already set up for us in the source image; instead we overwrite the entrypoint's parameters to call our `lambda_handler()` function inside the `lambda_function.py` script with the `CMD` command.

***WARNING:*** _Lambda requires that the output of `lambda_handler()` be serializable in order to convert it to JSON and send them back to the user. The code from previous sections returns numpy arrays of type `float32`, which are not serializable by the JSON library. This issue can easily be solved by converting a numpy array to a Python list with the `tolist()` method or by casting each `float32` value to Python floats with `float()`._

If you want to test the image, the URL for accessing the method in the Docker image is `http://localhost:8080/2015-03-31/functions/function/invocations`.

# Creating the Lambda function

Once you've generated a working Docker image, you can publish it to Lambda by selecting _Container image_ in the _Create function_ options. You must first upload the image to _Elastic Container Registry_ (ECR) and then copy the resulting URL and paste it in Lambda.

## Uploading to ECR (CLI)

You can install the AWS CLI tool with `pip install awscli`.

1. Create a registry in ECR
    * `aws ecr create-repository --repository-name my-registry`
    * The output of this command will contain the registry URI. Copy it. Take note of the `aws_account_id` (first part of the URI) and the `region` (string right before `amazonaws.com`)
1. Get the login info for the repo
    * (**DEPRECATED**, go to step 3 directly) `aws ecr get-login --no-include-email`
    * This will output a new command that you can use to login to the repo.
        * Your password will be visible in plain text. If you are sharing your screen, you can use `sed` to parse the text and change the output.
        * `aws ecr get-login --no-include-email | sed 's/[0-9a-zA-Z=]\{20,\}/PASSWORD/g'`
            * `sed` will parse the text and look for a string of length 20 containing numbers, upper and lowercase letters and the `=` sign, and replace it with the word `PASSWORD`.
1. Login to the repo. 2 ways:
    1. (**DEPRECATED**) by making use of the `get-login` info; 2 ways:
        1. Copy the output of the previous command.
        1. Use this trick to use the output of a command as the actual command:
            * `$(aws ecr get-login --no-include-email)`
    1. `aws ecr get-login-password --region` _region_ `| docker login --username AWS --password-stdin` _aws_account_id_`.dkr.ecr.`_region_`.amazonaws.com`
        * Make sure to change `region` and `aws_account_id` with the info you got from step 1.
1. Create the `REMOTE_URI` of your image by attaching a ***tag*** to the end of the repo URI preceded by a colon.
    * Consider the example URI `123456.dkr.ecr.eu-west-1.amazonaws.com/my-registry`.
        * `123456` is the ***account***.
        * `dkr.ecr` means that the URI belongs to an Amazon ECR private registry.
        * `eu-west-1` is the ***region***.
        * `amazonaws.com` is the top domain.
        * `/my-registry` is the directory of the registry we created in step 1.
    * A _tag_ is a name you assign to the specific version of the image in use. For our example we'll use `model-001`.
    * The resulting `REMOTE_URI` is `123456.dkr.ecr.eu-west-1.amazonaws.com/my-registry:model-001`
1. Tag your latest Docker image with the `REMOTE_URI`
    * `docker tag my-model-image:latest ${REMOTE_URI}`
1. Push the image
    * `docker push ${REMOTE_URI}`

## Finishing the setup on Lambda's Control Panel

Back in Lambda's Control Panel website, select _Container image_, give it a name and paste the `REMOTE_URI` into the _Container image URI_, or select it after clicking on _Browse images_ (if you do this, Amazon will paste a _digest_ of the image rather than the URI, but the end result is the same). Leave the default `x86_64` architecture as it is.

You may need to increase the timeout default value. You can do so on the _Configuration_ tab > _General configuration_ > _Edit_ button > _Timeout_. You may also need to increase the available RAM memory; you can do so from the same submenu as timeout.

Make sure to test your function with the _Test_ tab.

## Pricing

Lambda charges you each time that your `lambda_handler` method is called; specifically, it charges by millisecond of use. Pricing may vary depending on the chosen region and selected memory.

# Exposing the Lambda function (API Gateway)

***API Gateway*** is another AWS service for creating and exposing APIs.

1. Create a new REST API. Give it a name and create it. This will open a _Resources_ menu.
1. Create a new resource. Give it a name as well (usually something like `predict`) and create it.
1. Create a new _Method_ inside your resource. Choose the HTTP method you need (we'll use `POST` for this example). Select it to open its Setup options
    * Set _Integration type_ to _Lambda Function_.
    * Set _Lambda Region_ to your chosen region.
    * Choose your Lambda Function.
    * Use the Default Timeout and uncheck _Lambda Proxy integration_.
1. Test the gateway by clicking on _Test_.
    * Write a JSON document in _Request Body_ for testing.
1. After successful testing, deploy the API.
    1. Click on the _Actions_ button on top and choose Deploy API
    1. Assign a _Stage name_ such as `test`.
    1. After clicking on _Deploy_, the control panel will display a _Invoke URL_.
1. Test the deployment.
    * Copy the invoke URL and append the method you created previously at the end, like this: `https://woeijfw.execute-api.eu-west-1.amazonaws.com/test/predict`

