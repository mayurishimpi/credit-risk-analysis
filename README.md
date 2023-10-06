# credit-risk-analysis
One of the major activities of financial institutions such as banks is lending money to borrowers. It is very important to know that the borrowers are creditworthy. It is a challenging task to predict the risk while lending money. The decision needs to be made considering multiple factors including the current as well as the history of the financial status of the borrower. A system with high accuracy and optimality is needed for such decision-making. In this project, a survey of various classification algorithms is done and their performances are compared on a lending club dataset. The dataset consists of over 800,000 loan requests and 151 features. Dataset is first pre-processed to remove noise and extract the valuable features. Machine learning models with algorithms like Support Vector Machines, Random Forest, and Multi-layer Perceptron (MLP) are developed and tested on the dataset. The performance is measured using multiple metrics such as precision, recall, ROC, and AUC curves. The Random forest classifier generated a good accuracy score on both the training and testing phases, whereas other models like SVM did not perform as well.

# Dataset
The dataset consists of 800,000 loan records that "LendingClub" issued between the
years 2015 and 2018. The dataset under consideration is from the year 2016 (Q1) and has  133,886 loan records. It consists of 151 features to describe the loan application. The "Loan Status" indicates the present state of the loan with values, "Issued", "Current", "Fully paid", "Default", "Charged off', "Late (16-30 days)", "Late (31-120 days)" and "In grace period". These statuses are used to reduce them to a binary classification problem, i.e., the loan applications with "Charged off', "Default" are considered as "bad" or "defaulted" loans while "Fully Paid" is classified as "good" loans and remaining are ignored. The "Ioan status" attribute is replaced with "is bad" which takes the values of 0 for a good credit and 1 for a bad credit or default.

# Data Preparation
To clean the data, I dropped all features which contained more than 80% missing values, like “member_id” and “settlement_percentage”. I also dropped features that do not affect the classification and analysis for ML models, like “zip_code” and “loan_id”. The loan samples that had significantly high missing entries (>80%) were also dropped. The missing cells in the Numerical Features were filled in with the median entry in that column. Similarly, the missing entries in Categorical Features were filled by the mode of that column. Furthermore, the categorical features were converted into numerical and the features with multicollinearity were discarded.

# Data analysis
I plotted graphs to analyze trends in the data. Dataset was divided into two parts along with the percentage of the training data 80% and the testing data 20%.

# Evaluation metrics
As this is a binary classification problem, a ROC curve can be generated. For comparisonpurposes, performance metrics for SVM, Random Forest Model, and MLP model were computed, including accuracy, misclassification cost, precision, recall, and f1-measure. The performance of classification problems at various threshold settings is measured by the area under the curve (AUC) - ROC curve. ROC is a probability curve and AUC
represents the degree or measure of separability. AUC indicates how well a model predicts 0 classes as 0 and 1 classes as 1. The higher the AUC, the better the model.

# Model Parameters
● Random Forest
Sampling Strategy = random under-sampling, 0.5, Random States = 42,
Test size = 20%, Cross Validation = 5 fold

Model          Train Score Test Score Recall Score ROC
Random Forest  1.000       0.9996     0.0997       0.9998

● Support Vector Machine
Test size = 30%, Random states 42, Kernel function = linear, RBF, sigmoid,
polynomial, Cross-validation (cv) = 5 fold, balanced classification

Model Train Score Test Score Recall Score ROC
SVM   0.8547      0.6163     0.7338       0.5307


● Multi-Layer Perceptron
Test size = 30%, Hidden states = 2, Max iterations = 200, Cross Validation
= 5 fold
Model Train Score Test Score Recall Score ROC
MLP   0.8738      0.8688     0.9297       0.837


# CONCLUSION
After applying the algorithms on both the training and testing dataset, it seems that SVM does not work well (score is around 60%), this indicates that the classification is not optimal. In contrast, Random Forest produced significantly high accuracy scores on the train and test sets. For MLP, it is observed that by selecting 2 hidden layers and 200 max iterations, the test and train scores are good. Hyperparameter tuning is an important technique that chooses a set of optimal estimators from each algorithm that produce a higher accuracy score on the loan-level dataset. With more hyperparameter tuning, we can expect further optimization. It was observed that among the closed transactions, nearly 80% have good credit, and ML models like Random Forest classifier and MLP can be successfully used to evaluate credit risk on outstanding transactions for this dataset