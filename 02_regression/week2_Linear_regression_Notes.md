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
