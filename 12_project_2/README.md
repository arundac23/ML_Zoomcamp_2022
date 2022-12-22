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
