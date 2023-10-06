import pandas as pd
import numpy as np

import time

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, roc_curve, classification_report

data = pd.read_csv('data_linear.csv')
data.shape

pd.set_option('display.max_columns', None)
# print(data.head())

data = data.drop(['Unnamed: 0'], axis=1)
# print(data.describe())

Target = data['loan_status']
Features = data.drop(['loan_status'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.2, random_state=30)
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
y_test.fillna(y_test.mean(), inplace=True)
y_train.fillna(y_train.mean(), inplace=True)
KF = KFold(3, shuffle=True, random_state=30)
X1 = StandardScaler().fit_transform(X_train)
X2 = StandardScaler().fit_transform(X_test)

start1 = time.time()
Logit = SGDClassifier(loss='log', shuffle=True, n_jobs=-1,
                      warm_start=True, class_weight="balanced").fit(X1, y_train)
end1 = time.time()
Time1 = end1 - start1
print(Time1)

train_scores1 = cross_val_score(Logit, X1, y_train, cv=KF)
test_scores1 = cross_val_score(Logit, X2, y_test, cv=KF)

train_ac1 = round(Logit.score(X1, y_train), 4)
test_ac1 = round(Logit.score(X2, y_test), 4)
recall1 = round(recall_score(y_test, Logit.predict(X2)), 4)
roc1 = round(roc_auc_score(y_test, Logit.predict_proba(X2)[:, 1]), 4)

print(train_scores1)
print(test_scores1)
print(train_ac1)
print(test_ac1)
print(recall1)
print(roc1)

# SVM

from sklearn.kernel_approximation import Nystroem

X1_svm = Nystroem(random_state=42, n_jobs=-1, n_components=77, gamma=1 / 77).fit_transform(X1)
X2_svm = Nystroem(random_state=42, n_jobs=-1, n_components=77, gamma=1 / 77).fit_transform(X2)
from sklearn.calibration import CalibratedClassifierCV

start2 = time.time()
SVM = SGDClassifier(loss='hinge', shuffle=True, n_jobs=-1, random_state=42, warm_start=True)
calibrated_svm = CalibratedClassifierCV(SVM, method='isotonic').fit(X1_svm, y_train)
end2 = time.time()
Time2 = end2 - start2
print("SVM")
print(Time2)

train_scores2 = cross_val_score(calibrated_svm, X1_svm, y_train, cv=KF)
test_scores2 = cross_val_score(calibrated_svm, X2_svm, y_test, cv=KF)

train_ac2 = round(calibrated_svm.score(X1_svm, y_train), 4)
test_ac2 = round(calibrated_svm.score(X2_svm, y_test), 4)
recall2 = round(recall_score(y_test, calibrated_svm.predict(X2_svm)), 4)
roc2 = round(roc_auc_score(y_test, calibrated_svm.predict_proba(X2_svm)[:, 1]), 4)
# print(confusion_matrix(y_test, y_pred=y_train[1]))

fpr, tpr, thresholds = roc_curve(y_test, calibrated_svm.predict_proba(X2_svm)[:, 1])
plt.plot(fpr, tpr)
plt.title("SVM")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

print(train_scores2)
print(test_scores2)
print(train_ac2)
print(test_ac2)
print(recall2)
print(roc2)

print("Naive Bayes")
start3 = time.time()
GNB = GaussianNB().fit(X1, y_train)
end3 = time.time()
Time3 = end3 - start3
print(Time3)
train_scores3 = cross_val_score(GNB, X1, y_train, cv=KF)
test_scores3 = cross_val_score(GNB, X2, y_test, cv=KF)

train_ac3 = round(GNB.score(X1, y_train), 4)
test_ac3 = round(GNB.score(X2, y_test), 4)
recall3 = round(recall_score(y_test, GNB.predict(X2)), 4)
roc3 = round(roc_auc_score(y_test, GNB.predict_proba(X2)[:, 1]), 4)

print(train_scores3)
print(test_scores3)
print(train_ac3)
print(test_ac3)
print(recall3)
print(roc3)

fpr, tpr, thresholds = roc_curve(y_test, GNB.predict_proba(X2)[:, 1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

df = pd.DataFrame([['Logit', train_ac1, test_ac1, recall1, roc1],
                   ['SVM', train_ac2, test_ac2, recall2, roc2],
                   ['Naive Bayes', train_ac3, test_ac3, recall3, roc3]],
                  columns=['model', 'train score', 'test score', 'recall score', 'ROC AUC score'])
print(df)
df.to_csv('result.csv')
