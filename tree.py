import pandas as pd
import numpy as np

import time

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, roc_curve, classification_report
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv('data_non_linear.csv')
print(data.shape)
data['loan_status'].value_counts()
pd.set_option('display.max_columns', None)
# print(data.head())

Target = data['loan_status']
Features = data.drop(['loan_status'], axis=1)

# Under sampling the imbalanced dataset

undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
features_balanced, target_balanced = undersample.fit_resample(Features, Target)
print(features_balanced.shape)
print(target_balanced.shape)
X_train, X_test, y_train, y_test = train_test_split(features_balanced, target_balanced, test_size=0.2, random_state=42)
KF = KFold(5, shuffle=True, random_state=42)
# Random Forest
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
y_test.fillna(y_test.mean(), inplace=True)
y_train.fillna(y_train.mean(), inplace=True)

start1 = time.time()
forest = RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=42, class_weight="balanced").fit(X_train,
                                                                                                          y_train)
end1 = time.time()
Time1 = end1 - start1
print(Time1)

train_scores1 = cross_val_score(forest, X_train, y_train, cv=KF)
test_scores1 = cross_val_score(forest, X_test, y_test, cv=KF)

train_ac1 = round(forest.score(X_train, y_train), 4)
test_ac1 = round(forest.score(X_test, y_test), 4)
recall1 = round(recall_score(y_test, forest.predict(X_test)), 4)
roc1 = round(roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1]), 4)

fpr, tpr, thresholds = roc_curve(y_test, forest.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr)
plt.title('Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print((train_scores1))
print((test_scores1))
print((train_ac1))
print((test_ac1))
print((recall1))
print((roc1))

print(forest.score(X_train, y_train))
print("Confusion matrix, RF")
print(confusion_matrix(y_test, forest.predict(X_test)))
print(classification_report(y_test, forest.predict(X_test)))

print("MLP")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,2), random_state=1, max_iter=500).fit(X_train, y_train)

train_scores2 = cross_val_score(clf, X_train, y_train, cv=KF)
test_scores2 = cross_val_score(clf, X_test, y_test, cv=KF)

train_ac2 = round(clf.score(X_train, y_train), 4)
test_ac2 = round(clf.score(X_test, y_test), 4)
recall2 = round(recall_score(y_test, clf.predict(X_test)), 4)
roc2 = round(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]), 4)

print((train_scores2))
print((test_scores2))
print((train_ac2))
print((test_ac2))
print((recall2))
print((roc2))

print(clf.score(X_train, y_train))
print("Confusion matrix, MLP")
print(confusion_matrix(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test)))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr)
plt.title('MLP')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
df = pd.DataFrame([['Random Forest', train_ac1, test_ac1, recall1, roc1],
                  ['MLP', train_ac2, test_ac2, recall2, roc2]],
                 columns=['model', 'train score','test score','recall score', 'ROC AUC score'])
print(df)
