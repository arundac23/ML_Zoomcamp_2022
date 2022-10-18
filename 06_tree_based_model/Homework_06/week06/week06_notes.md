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
### Decision tree learning algorithm:
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
