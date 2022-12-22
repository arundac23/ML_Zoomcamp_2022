# Image Classification of landscape scenes:

This project work is based on [Alexey Grigorev's](https://github.com/alexeygrigorev) course #MLZOOMCAMP. The original repository of this course is located [here](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp)

## Dataset details:
This is image data of natural Scenes around the world.
`The original dataset is available here` [dataset link](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

`This Data contains around 25k images of size 150x150 distributed under 6 categories.

### ['buildings','forest','glacier','mountain','sea','street']

The Train, val, test and Prediction data is separated in folders. There are around 14k images in Train, 3k in validation and 64 in test and 7k in pred folder.

## Project Description:

For this capstone project_1, I decided to work on image classification using deep learning cnn architecture.Since I am a complete beginner in deep learning concepts, I took a simple image classification task for this project. Due to time constraint i am not able to explore all other concepts or tools of deep learning during this project. I just followed the same techniques that were taught during the MLzoompcamp deep learning session. Hoping to explore different concepts during the upcoming capstone project_2.

For this project, I didn't train this images from scratch.Since it requires large set of images to improve accuracy. I used transfer learning techinique for image feature extraction.

2 different convolutional architectures were used:

* `Xception`: one of the most popular application available on Keras.
* `inception_v3`: It is the third edition of Inception CNN model by Google. It is computationally less expensive. It uses auxiliary Classifiers as regularizes.

## Files:

`README.md`     : Full description about this project.

`Notebook.ipynb`: The jupyter notebook that covers image preparation, model training, hyperparameter tuning, model selection and final model saved in H5 format and    conversion into Tflite format.

train.py: Converted python file from Notebook.ipynb for the training of selected model and saving the final model for deployment.

`model` folder : Xception tflite model(xception_model.tflite) and trained `h5` model of both Xception(xception_v_1_11_0.930.h5) and inceptionv3(inception_v3_1_10_0.927.h5) is saved in this folder.

`conda_env_requirements.txt` : Text file for installing all required dependencies in Conda environment.

`Dockerfile` : The file necessary for the docker image creation.

`lambda_function.py`: python app file for deployed model predictions. This file is required deployment on AWS Lambda serverless service.

`test.py`: The python file test the response from containerized version in Docker.

`test_aws_lambda.py`: The python file test the response from AWS lambda server.

## Environment setup:
I have used WSL2 for my model testing and deployment.All the codes were ran and tested through `ubuntu 22.04.1 LTS`.I have used conda for environment setup

Create an environment by using `conda_env_requirements.txt`:

    * `conda create --name <env_name> --file conda_env_requirements.txt` 
## Building the docker image:

Before creating the Docker image, please make sure that the `Dockerfile`, `xception_model.tflite` and lambda-function.py files are in the folder
`docker build -f Dockerfile -t landscape-image-classifier .`

Please note the period at the end of the command. You may change `landscape-image-classifier` to any other name.

Once the image has been created, run it with the following command:

`docker run -it --rm -p 8080:8080 landscape-image-classifier:latest`

open a another terminal and run the test.py to see the prediction response

Run `python test.py`

## Model deployment in AWS Lambda:
Model was deployed in AWS lambda. 

## Pushing Docker image to AWS:
Install and configure AWS CLI using secret key in your terminal. Then login into AWS service and look for elastic container registry.

Then click create repository

![image](https://user-images.githubusercontent.com/76126029/209187794-24d87764-a629-4d68-84b9-80eb8a5d8384.png)

Give some name to repository(Example : landscape-image-classifier-tflite)

![image](https://user-images.githubusercontent.com/76126029/209188051-a377cc84-5964-42b5-978e-b8edc92663da.png)

Once repository is created. Look for `view push command`

![image](https://user-images.githubusercontent.com/76126029/209188484-57df8844-6f07-414f-a8c9-8092b23e0b2b.png)

 Authenticate your Docker client to your registry- Run the similar authentication command in your terminal and provide your credentials
 
 Example : `aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 029243105711.dkr.ecr.us-east-2.amazonaws.com`
 
 After you build the docker image, tag your image so you can push the image to this repository:
 
 Example : `docker tag <your docker image name>:<tag id> 029243105711.dkr.ecr.us-east-2.amazonaws.com/landscape-image-classifier-tflite:latest`
 
Finally Run the following command in terminal to push this image to your newly created AWS repository:

Example : `docker push 029243105711.dkr.ecr.us-east-2.amazonaws.com/landscape-image-classifier-tflite:latest`

## Set up a Lambda function in AWS.
* Open the Lambda dashboard on AWS and click on Create function.

Select `container image` option and enter the `function-name` based on the model (`Example: image-classifier). Keep other setting as default.

![image](https://user-images.githubusercontent.com/76126029/209191734-f48293ff-7843-420d-b89a-606ead4afcf0.png)

Then `browse images` to select the already pushed docker image.

![image](https://user-images.githubusercontent.com/76126029/209192290-a041dd20-1dea-475b-8ac2-2619bfcb48d5.png)

create a function. Your page should look like this

![image](https://user-images.githubusercontent.com/76126029/209192795-4d61eab7-fa56-424a-9f72-d0146f06f7df.png)

Then click on test tab. Click `create new event` --> Enter `Event name` --> Enter request URL in event JSON

![image](https://user-images.githubusercontent.com/76126029/209193486-0d81df5b-9b4a-43a0-bb04-bdc5c0e61e6d.png)

Then go to `confirguration tab` --> edit --> increase memory to 1024 mb and timeout to 30s.

![image](https://user-images.githubusercontent.com/76126029/209194424-ccbc349e-f019-4216-8471-477eb3872639.png)

Comeback to test tab > click the test button --> you should see the response based on your url request

![image](https://user-images.githubusercontent.com/76126029/209194940-875fc03c-1294-4c7a-85c5-8dd4aa2e7c2a.png)

## Exposing lambda function to API gateway:

Look for API gatway in AWS. Then select `REST API`

![image](https://user-images.githubusercontent.com/76126029/209195448-3b832406-d67c-499f-93e2-4c6ba7e74d2d.png)

create new API and integrete to AWS lambda function.

![image](https://user-images.githubusercontent.com/76126029/209195645-c1c5416e-5361-4b1d-b6bd-c70a8714320c.png)

After integreting the AWS lambda function, test the service by clicking the test button

![image](https://user-images.githubusercontent.com/76126029/209196157-b7fa19ac-71d1-44af-9943-1f44cd9e2fa0.png)

if testing is ok, you should see the response for your Url request.

![image](https://user-images.githubusercontent.com/76126029/209184761-b0a05fe8-263d-42e9-b634-9d8778fad7b6.png)

Once response is ok, you can deploy it using `Actions` --> `create deploy`

![image](https://user-images.githubusercontent.com/76126029/209196723-c1c4e3d7-41e3-436a-a06c-182fe32ec3bd.png)

Final deployment page should be like this. You check the final prediction using following API

![image](https://user-images.githubusercontent.com/76126029/209181059-0c206b1b-a3c5-441f-b7e4-d6f5a2069730.png)

Use run `test_aws_lambda.py` in your terminal to see the response for image classification based on your 'url' request.

Change the url of other natural scene to see the final prediction.
![image](https://user-images.githubusercontent.com/76126029/209182988-5c4cc009-868b-4d26-ac5e-f0a28135efac.png)



