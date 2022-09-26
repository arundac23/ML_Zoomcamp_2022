# Logistic Regression - Binary classification
Aim of this project is to learn classification problem through telco churn data. Using classification algorithm, we will separate customer who goes out of that network as 1 and the one who stays with same network as zero.
In this case, we will identify each customer score, based on that we will classify that into two categories.
* `g (x<sub>i</sub>) = (y<sub>i</sub>)

In this formula, (y<sub>i</sub>) is predicting the chance of being {0,1}. If model predicts 0, then customers is not churning and if it is 1 then they are ready to churn.
# Data Preparation
In the data preparation section, data was loaded and organized in the structured way. Following commands are used in this section
* `pd.read.csv()` - read csv files 
* `df.head()` - take a look at top 5 rows of the dataset 
* `df.head().T` - transposed look of df.head()
* `df.columns` - show the columns names of the dataset 
* `df.columns.str.lower()` - converts columns names into lower cases
* `df.columns.str.replace(' ', '_')` - replace the space separator with underscore 
* `df.dtypes` - returns data types of all the columns 
* `df.index` - return indices of a dataframe
* `pd.to_numeric()` - convert a object values to numerical values. 
* The `errors=coerce` allows that changes even if it is encountered with errors. 
* `df.fillna()` - replace NAs with some value 
* `(df.x == "yes").astype(int)` - convert x booleen values to numerical values.
## Setting up the validation framework
Performing the train/validation/test split with Scikit-Learn
following module is imported for data split
from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
Target variable should be created using Dataframe and it should be removed from the feature matrix.
##  EDA
The EDA for this project consisted of: 
* Checking missing values 
* Looking at the distribution of the target variable (churn)
* Looking at numerical and categorical variables 
* `df.isnull().sum()` - returns the number of null values in the dataframe.  
* `df.x.value_counts()` returns the number of values for each category in x series. The `normalize=True` argument retrieves the percentage of each category. In this project, the mean of churn is equal to the churn rate obtained with the value_counts method. 
* `round(x, y)` - round an x number with y decimal places
* `df[x].nunique()` - returns the number of unique values in x series 
## Feature importance: Churn rate and risk ratio
chrun rate : identify the crictical feature that increase the chance of chruning rate.
 Difference between mean of the target variable and mean of categories for a feature. If the difference is large, then that particular feature has more influence on the outcome of the results.
 Risk ratio : It is ration between feature category mean and global mean of target variable.If ratio is greater than 1, then there is high possibility of chruning and vice versa if ratio is less than 1.
 ## Feature Importance: Mutual Information
