# 2. Machine Learning Zoomcamp- Week 2
Thanks to Alexey Grigorev for organizing this free ML course. Main repository of this course is available here [course link](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp)
## 2.1 Car Price Prediction Project
This week linear regression topic is covered using car Price Prediction Project. Scope of this project to create a linear regression model to predict the best car price for users based on its features.The code and dataset is available in this [link](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/data.csv).

## Project plan:
* Prepare data and Exploratory data analysis (EDA)
* Use linear regression for predicting price
* Understanding the internals of linear regression
* Evaluating the model with RMSE
* Feature engineering
* Regularization
* Using the model

[Car Project Notebook link](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/02-regression/notebook.ipynb)

## 2.2 Data Preparation
Data preparation is first step of the Machine learning process. In this section, Numpy and Pandas library were mainly used to load and normalize the data.
Following methods or commands are used to read and manipulate the data
* df = pd.read_csv('data.csv') --> For reading the data in the CSV format
* df.head() --> returns the top n rows of data based on the request.
* df.columns - retrieve colum names of a dataframe 
* df.columns.str.lower() - convert column name to lowercase letters.
* df.columns.str.replace(' ', '_') - replace the columns name with space to underscore
* df.dtypes - returns data types of all features 
* df.index - returns indices of a dataframe

## 2.3 EDA
EDA is next step that refers to be critical process to perform initial study on data to identify patterns, duplicates in the data and check for assumption based on its statistical information and visualization plots. In addition to Numpy and Pandas python packages, Matplotlib and seaborn libraries are used in this section for visualization purpose.
* df[col].unique() --> returns the unique values in that column
* df[col].nunique() --> returns the number of unique values in that column
* df.isnull().sum() --> identies the number of NA values
* sns.histplot(df.msrp, bins=50) --> histogram plot of the column or series `msrp`.
* np.log1p() --> log transformaion of the variable with addition of `1` . 
* If original data has long head or tail, then it causes confusion in ML prediction. `Log transformation` is mainly used to transform the skewed data to approximately similar to normal distribution.
## 2.4 Setting up the validation framework
Before implementing the machine learning algorithms. Dataset should be divided into three parts 1. Training 2.Validation and 3.Test. For each dataset, X features variables in matrix form and y target variables in Vector form need to be extracted. All the dataset should be shuffled using random module and shuffled indices is used for dataset partition.
* n = len(df)
* n_val = int(n * 0.2)
* n_test = int(n * 0.2)
* n_train = n - n_val - n_test --> Dividing the dataset into 3 parts with 60% for training and 20% each for validation and test dataset
* idx = np.arange(n)
* np.random.seed(2)
* np.random.shuffle(idx) --> Shuffle the ids using randon seed and shuffle method
* df_train = df.iloc[idx[:n_train]]
* df_val = df.iloc[idx[n_train:n_train+n_val]]
* df_test = df.iloc[idx[n_train+n_val:]] --> Assign the shuffled ids for dataset partition.
## 2.5 Linear regression
### General formula for ML
* `g(X) ≈ y`
* `g` = regression model
* `X` = feature matrix
* `y` = target

Linear regression model is a function to predict the target variable `y` based on the input matrix `X`. End goal is predicting `y` should be closer to the actual 'y'. In the car project, Car price should be predicted based on the car features.
Let consider three car features to predict the price of the car

Formula for appling the linear regression `g(xi)` is 
* `g(Xᵢ) = w₀ + w₁·xᵢ₁ + w₂·xᵢ₂ + w₃·xᵢ₃`
Alternatively:
* `g(Xᵢ) = w₀ + ∑( wⱼ·xᵢⱼ, Limit j= 1 --> 3`
* `w₀` is the bias weight term.
* All other `wⱼ` are weights for each features.
## 2.6 Linear Regression in vector form
* `g(Xᵢ) = w₀ + Xᵢᵀ · W`
* `Xᵢᵀ` is the transposed feature matrix.
* `W` is the weights.
Linear regression is calculated from dot product of feature matrix and vector of weights. Dot product should be approximately equal target `y`.
Xᵢᵀ · W ≈ y
## 2.7 Normal equation
Solving of bias term W from the following formula.
* `g(X) = X·W ≈ y` --> 1
* Multiply this term with Xᵀ on the both sides
* `Xᵀ.X·W ≈ Xᵀ y` --> 2
* Xᵀ.X is gram matrix. It is a square matrix, inverese is always possible.
* Now multiply the inverse term of gram matrix on both side of the equation 2
* `(Xᵀ · X)⁻¹ Xᵀ.X·W ≈ (Xᵀ · X)⁻¹ Xᵀ y`
* The term `(Xᵀ · X)⁻¹ Xᵀ.X` become identity matrix `I` and considered as 1
So the final equation is 
`W ≈ (Xᵀ · X)⁻¹ Xᵀ y`
## 2.8 Baseline model for car prediction project
In the car prediction project, % features was selected for predicting the price. But one of the feature `engine HP` has some missing value. That feature is filled with zero at the missing location before implementing the linear regression model.
## 2.9 RMSE (Root Mean Square Error)
RMSE is one of the method to measure the accuracy of the prediction model with repest to actual target.
* `RMSE = √( 1/m * ∑( (g(Xᵢ) - yᵢ)²	 , Limit i= (1 --> m))`
* `g(Xᵢ)` is the prediction for `Xᵢ`.
* `yᵢ` is the actual value.
* If RMSE value is low, then the model is considered as high accurate model in prediction.
## 2.10 computing RMSE in validation dataset
Same RMSE formula should be applied for Validation dataset to ensure the model accuracy is good.
## 2.11 Feature Engineering
Feature engineering is adding a new key feature to improve the prediction of the model. In the car price prediction project, age is of the car is the critical feature that vary the price of the car. So  `age of the car` is added as additional feature. Thus reduces RMSE value and improves the model prediction.
## 2.12 Categorical variables
Categorical variables is distinct features that be classified into different categories. In the car example project, Make, model, engine_fuel_type, transmission_types, Driven_wheels, vehicle_style. These categorical features can be added to the existing feature lists to see the effect in regression model prediction. 
## 2.13 Regularization
Sometimes the Gram matrix `Xᵀ · X` becomes a singular matrix and it cannot be inverted.This could happened, when feature matrix `X` has duplicate features. 
When data is not super clean,data may have almost identical features, the Gram matrix is inverese is possible in that situation but the end result values would be huge. This affects the ML model prediction. Since it increases the weightage `w` values.

Regularization helps to reduce to the weightage value in this situation. Regularization parameter is applied to the diagonal of the Gram matrix. 
The regularization parameter is usually a small decimal value such as `0.00001` or `0.01`. The larger the parameter, the smaller the final values in the inverted matrix. Sometimes big regularization parameter can lead to worse performance than smaller parameters. Model tuning is required to get the optimal value.

It can easily implemented using numpy by creating an identity matrix with `np.eye()`, multiplying it with our regularization parameter and finally adding the final result to the Gram matrix.
## 2.14 Tuning the model
As i mentioned before, model need to be tuned with different regularization values to find the optimal parameter to imporve the model prediction.
## 2.15 Using the model
After improving the model based on validation dataset. Finally both training and validation should be combined and make the final prediction based on test database.
Training and validation dataframe can be combined using
df_full_train = pd.concat([df_train, df_val])
