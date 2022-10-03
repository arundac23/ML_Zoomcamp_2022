## Binary Classifcation evaluation metrics
Fourth week of Machine Learning Zoomcamp organized by [alexeygrigorev](https://github.com/alexeygrigorev). This week covers the different metrics to evaluate a binary classifier that includes accuracy, confusion table, precision, recall, ROC curves(TPR, FRP, random model, and ideal model), AUROC, and cross-validation
## Accuracy and Dummy Model:
In a binary classification model, it is possible to predict the model as positive which turn out to be actual negative. Similarly the negative prediction may turn out be positive as well.
Accuracy is ratio of correct prediction to  overall predictions.
Accuracy = True Positive & true Negative prediction / Total prediction

In the churn rate problem, we decided threshold as 0.5 for classification. It should not be always 0.5. But, in this particular problem, the best decision cutoff of 0.5  has hightest accuracy 80%

But if we build a dummy model in which the decision cutoff is 1, so the algorithm predicts that no clients will churn, the accuracy would be 73%. Thus, we can see that the improvement of the original model with respect to the dummy model is not as high as we would expect.

Therefore, in this problem accuracy can not tell us how good is the model because the dataset is unbalanced, which means that there are more instances from one category than the other. This is also known as class imbalance.

### Confusion table
Confusion table is a way of measuring technique to identify the errors and correct decisions that binary classifiers can make. It is helpful to evaluate the model classification.

when `Predicted positive is actual positive` :arrow_right: `True Positive`

`predicted positive is actual negative` :arrow_right: `False Positive`

`predicted negative is actual positive` :arrow_right: `False Negative` 

`predicted negative is actual negative` :arrow_right: `True Negative`

|**Actual :arrow_down:     Predictions:arrow_right:**|**Negative**|**Positive**|
|:-:|---|---|
|**Negative**|TN|FP|
|**Postive**|FN|TP| 

***Precision*** is the _fraction of positive predictions that are correct_. We only look at the positive predictions, and we calculate the fraction of correct predictions in that subset.

* `precision = TP / (TP + FP)`

***Recall*** is the _fraction of correctly identified positive examples_. Instead of looking at all of the positive predictions, we look at the _ground truth positives_ and we calculate the fraction of correctly identified positives in that subset.

* `recall = TP / (TP + FN)`

In the churn rate problem, the precision and recall values were only 67% and 54% when compared to accuracy of 80% . Thus concludes that these measures reflect some errors of our model that accuracy did not show due to the class imbalance.

### ROC Curves
ROC stands for Receiver Operating Characteristic curve is a graphical plot used to show the diagnostic ability of binary classifiersThis measure considers False Positive Rate (FPR) and True Postive Rate (TPR), which are derived from the values of the confusion matrix.

FPR is the fraction of false positives (FP) divided by the total number of negatives. Formula for FPR is
* `FPR = FP / (TN + FP)`
we need to maximize this FPR matrics

TPR or Recall is the fraction of true positives (TP) divided by the total number of positives. Formula for TPR is
* `TPR = TP / (TP + FN)`
Tpr need to be minimized.

![image](https://user-images.githubusercontent.com/76126029/193704408-2007ac2a-29f4-4a68-8acf-a9d112c4d5a3.png)


ROC curve considers both FPR and TPR at different threshold value of 0 or 1.Classifiers that give curves closer to the top-left corner indicate a better performance. As a baseline, a random classifier is expected to give points lying along the diagonal (FPR = TPR). The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.
Note that the ROC does not depend on the class distribution. This makes it useful for evaluating classifiers
### ROC AUC
The ROC AUC is the `area under the ROC curve`. The ROC AUC is a great metric of measuring performance
The AUROC of a random model is 0.5, while for an ideal one is 1.
### Cross Validation 
Cross validation (CV) is one of the technique used to test the effectiveness of a machine learning models, it is also a re-sampling procedure used to evaluate a model.

Split the entire data randomly into K folds (value of K shouldn’t be too small or too high, ideally we choose 3 to 10 depending on the data size). The higher value of K leads to less biased model (but large variance might lead to over-fit), where as the lower value of K is similar to the train-test split approach we saw before.

Then fit the model using the K-1 (K minus 1) folds and validate the model using the remaining Kth fold. Note down the scores/errors.

Repeat this process until every K-fold serve as the test set. Then take the average of your recorded scores. That will be the performance metric for the model.
