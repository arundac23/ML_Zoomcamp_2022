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
