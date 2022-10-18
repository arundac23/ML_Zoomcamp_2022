## Decision trees Ensemble learning
In this session we'll learn about decision trees and ensemble learning algorithms. The questions that we try to address this week are, "What are decision trees? How are they different from ensemble algorithms? How can we implement and fine-tune these models to make binary classification predictions?"
We will use [credit scoring data](https://github.com/gastonstat/CreditScoring) to explain the concepts.
In this credit scoring classification problem, 
- if the model returns 0, this means, the client is very likely to payback the loan and the bank will approve the loan.  
- if the model returns 1, then the client is considered as a `defaulter` and the bank may not approval the loan.
## Decision trees
The intuition behind Decision Trees is that you use the dataset features to create yes/no questions and continually split the dataset until you isolate all data points belonging to each class. It is organized in tree structure.
Every time you ask a question you’re adding a node to the tree. And the first node is called the root node.
If you decide to stop the process after a split, the last nodes created are called leaf nodes.
![image](https://user-images.githubusercontent.com/76126029/196307942-5dffda2d-aff0-438c-82cb-f12453829a04.png)
With versatility, the decision tree is also prone to overfitting. One of the reason why this algorithm often overfits because of its depth. It tends to memorize all the patterns in the train data but struggle to performs well on the unseen data (validation or test set).

To overcome with overfitting problem, we can reduce the complexity of the algorithm by reducing the depth size.
The decision tree with only a single depth is called decision stump and it only has one split from the root.
**Classes, functions, and methods**:
* `DecisionTreeClassifier().fit(x,y)` - Scikit-Learn class to train a decision tree classifier with x feature matrix and y target variable.
* max_depth --> parameter to define the depth of the tree.
* `export_text(x, feature_names=y)` - Scikit-Learn class to print x tree with y feature names.

## Decision tree learning algorithm:
A decision tree is made of decision nodes that can be true or false, taking into account if a the value of a feature is greater or equal to a threshold.
Starting with a simple dataset:

| index | assets | status |
| --- | --- | --- |
| 0 | 0 | default |
| 1 | 2000 | default |
| 2 | 3000 | default |
| 3 | 4000 | ok |
| 4 | 5000 | ok |
| 5 | 5000 | ok |
| 6 | 8000 | default |
| 7 | 9000 | ok |

We will create a tree with a single layer trained on assets. The purpose we have is to identify the best threshold value. It will split dataset into two parts based on threshold(T) value.

If we count every single unique value, we can define decision thresholds based on those values: `[0, 2000, 3000, 4000, 5000, 8000]`, with `assets > threshold` as our condition statement

If we set our threshold at `T=4000` we will split the data s below, [0,3] is split as left and [4,7] is splited as right

* assets > 4000 = ok

| index | assets | status | prediction |
| --- | --- | --- | --- |
| 0 | 0 | default | default |
| 1 | 2000 | default | default |
| 2 | 3000 | default | default |
| 3 | 4000 | ok | default |

* 3 correct prediction for left class, 1 wrong.

| index | assets | status | prediction |
| --- | --- | --- | --- |
| 4 | 5000 | ok | ok |
| 5 | 5000 | ok | ok |
| 6 | 8000 | default | ok |
| 7 | 9000 | ok | ok |

* 3 correct prediction for right class, 1 wrong.

We can now calculate the `impurity` for each threshold. We will use the ***misclassification rate*** for calculating the impurity. Misclassification rate is a method to calculate impurity, which measures proportion of predictions errors with respect to the expected decision.

* Impurity for left: `1 error / 4 total results = 1/4 = 0.25`
* Impurity for right: `1 error / 4 total results = 1/4 = 0.25`
* Average impurity: `0.25`

We can now calculate the impurity for all thresholds:

| Threshold | Decision left | Impurity left | Decision right | Impurity right | Average |
| --- | --- | --- | --- | --- | --- |
| 0 | default | 0% | ok | 43% | 21% |
| 2000 | default | 0% | ok | 33% | 16% |
| 3000 | default | 0% | ok | 20% | 10% |
| 4000 | default | 25% | ok | 25% | 25% |
| 5000 | default | 50% | ok | 50% | 50% |
| 8000 | default | 43% | ok | 0% | 21% |

Thus, the best threshold for this example would be `3000`, with an impurity of only 10%. 
In brief, finding the best split algorithm can be summarized as follows:

* FOR F IN FEATURES:
    * FIND ALL THRESHOLDS FOR F:
        * FOR T IN THRESHOLDS:
            * SPLIT DATASET USING "F>T" CONDITION
                * COMPUTE IMPURITY OF THIS SPLIT
* SELECT THE CONDITION WITH THE LOWEST IMPURITY

The algorithm can iterate until it completes all possible splits, but we can stablish a stopping criteria to define when we need to stop the splitting process. The stopping condition can be defined considering criteria such as:

* Group already pure
* Tree reached depth limit
* Group too small to split

Decision tree learning algorithm can be summarized as:

1. Find the best split
2. Stop if max-depth is reached
3. If left is sufficiently large and not pure
    * Repeat for left
4. If right is sufficiently large and not pure
    * Repeat for right

Decision trees can be used for regression tasks.

**Libraries, classes and methods:**

* `df.sort_values('x')` - sort values of an x column from a df dataframe.
* `display()` - IPython.display class to print values inside of a for-loop.

##  Decision trees parameter tuning

Decision trees have multiple parameters, such as **max depth**, which defines how deep the tree can grow or number of layers in the tree. Another parameter is the **minimum size of a group**, a criteria to decide if one side or group of a tree is sufficiently large. Decision trees have more parameters, but in this session we will cover only the two mentioned before.

Parameter tuning means to choose parameters in such a way that model's performance or evaluation metric is maximized or minimized, depending on what metric we choose. For this project, we will try to maximize AUROC metric.

A good strategy to tune various parameters in large datasets is first find the best values from one parameter, and then combine them with other values from other parameters.

**Classes and methods:**

* `DecisionTreeClassifier(max_depth=z, min_samples_leaf=w).fit(x,y)` - Scikit-Learn class to train a decision tree classifier with x feature matrix and y target values, using z as the maximum depth and w as minimum size of the left group.
* `df.pivot(index='x', columns='y', values='z')` - pandas method to transform the structure of a dataframe df considering x as rows, y as columns, and z as cell values.
* `sns.heatmap(df, annot=True, fmt='.zf')` - seaborn method to create a heatmap from a df dataframe with annotations and z rounding values.


##  Ensemble learning and random forest

In general, ensemble models use various models and aggregate their results with some operator such as mean, maximum, or another one. So, prediction of ensemble models is the aggregated operator of results from all models. RF is an ensemble of independent decision trees, and each of these models gets a random subset of features.

There is a point at which RF does not improve its performance although we increase number of trees in the ensemble. We also fine-tune **max depth**,and **minimum size of a group** because RF is a bunch of decision trees.

Other interesting parameters to tune are **the maximum number of features** and **bootstrap**, which is a different way of randomization at the row level. Also, **n_jobs** parameter allows us to train models with parallel processing.

**Classes and methods:**

* `RandomForestClassifier(n_estimators=z, random_state=w).fit(x,y)` - Scikit-Learn class to train a random forest classifier with x feature matrix and y target values, using z decision trees and a random seed of w.
* `dt.predict_proba(x)[:,1]` - predict x values with a dt Scikit-Learn model, and extracts only the positive values.
* `zip(x, y)` - function that takes x and y iterable or containers and returns a single iterator object, having mapped values from all the containers.


## Gradient boosting and XGBoost

Boosting algorithms are a type of ensemble models in which each model takes as input results of previous models in a sequential manner. At each step of boosting, the aim is to reduce errors of previous models. Final prediction of boosting algorithms consider predictions of all models using an aggregated operator. When models in boosting are decision trees, we called it as **Gradient boosting trees (GBT)**.

A good library for working with GBT is **xgboost**. Some important parameters of GBT from xgboost are:

* **eta:** learning rate, which indicates how fast our model learns
* **max_depth:** depth size of trees
* **min_child_weight:** how many observation we have in a leave node
* **objective:** type of model we will apply
* **eval_metric:** specify evaluation metric to use
* **nthread:** parallelized training specifications
* **seed:** random seed
* **verbosity:** type of warnings to show

**Libraries, classes and methods:**

* `xgboost` - Python library  that implements machine learning algorithms under the Gradient Boosting framework.
`xgb.DMatrix(x, label=y, feature_names=z)` - xgboost method for converting data into DMatrix, a special structure for working on training data with this library. We need to provide an x feature matrix, a y target vector, and a z vector of features names.
* `xgb.train(x, dtrain, num_boost_round=y, evals=z)` - xgboost method for training a dtrain DMatrix with x dictionary of parameters, y number of trees, and a z watchlist to supervise model performance during training.
* `x.predict(y)` - method from an x xgboost model to make predictions on a y validation dataset.
* `%%capture` - Jupyter notebook command to capture everything a code cell outputs into a string.
* `s.split('x')` - string method to split it by x separator.
* `s.strip('x')` - string method to delete some x characters.


##  XGBoost parameter tuning

**Learning rate** or **eta** refers to size of step in **Gradient boosting trees (GBT)**, and it tell us how fast a model learns. In other words, eta controls how much weight is applied to correct predictions of a previous model.

It is recommended to tune first eta parameter, then max_depth, and min_child_weight at the end. Other important parameters to consider for tuning are **subsample**, **lambda**, and **alpha**.


##  Selecting the best model

The best tree based model for credit risk scoring project was **Gradient boosting trees (GBT)**. So, we trained this model on the entire training dataset, and evaluated it on test dataset.

Usually, GBT is one of the models with the best performance for tabular data (dataframe with features). Some downsides fo GBT models are their complexity, difficulty for tunning, and tendency to overfitting.


## Summary

* Decision trees have nodes with conditions, where we split the dataset and we have decision nodes with leaves when we take a decision. These models learn if-then-else rules from data, and they tend to overfit if we do not restrict them in terms of depth growing.
* For finding the best split of decision trees, we select the least impure split. This algorithm can overfit, that's why we control it by limiting the max depth and the size of the group.
* Random forest is a way of combining multiple decision trees in which each model is trained independently with a random subset of features. It should have a diverse set of models to make good predictions.
* Gradient boosting trains model sequentially: each model tries to fix errors of the previous model. XGBoost is an implementation of gradient boosting.
