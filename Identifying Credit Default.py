import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import pandas_profiling as pp
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


df = pd.read_csv("UCI_Credit_Card.csv", index_col=0, na_values='')

X = df.copy()
y = X.pop('default.payment.next.month')

TEST_SIZE = 0.2

print(df.info())

print(df.describe().transpose().round(2))

msno.matrix(X)
plt.show()

report = pp.ProfileReport(df)
report.to_file('profile_report.html')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, gamma=0.7, subsample = 0.8, max_depth=6, min_child_weight=3, random_state=0)
xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)

accuracy = metrics.accuracy_score(y_pred, y_test)
recall = metrics.recall_score(y_pred, y_test)
precision = metrics.precision_score(y_pred, y_test)
f1 = metrics.f1_score(y_pred, y_test)

print(f"XGBoost accuracy score: {accuracy:.4f}")
print(f"XGBoost recall score: {recall:.4f}")
print(f"XGBoost precision score: {precision:.4f}")
print(f"XGBoost f1 score: {f1:.4f}")

lgb_clf = LGBMClassifier()
lgb_clf.fit(X_train, y_train)
y_pred = lgb_clf.predict(X_test)

accuracy = metrics.accuracy_score(y_pred, y_test)
recall = metrics.recall_score(y_pred, y_test)
precision = metrics.precision_score(y_pred, y_test)
f1 = metrics.f1_score(y_pred, y_test)

print(f"LightGBM accuracy score: {accuracy:.4f}")
print(f"LightGBM recall score: {recall:.4f}")
print(f"LightGBM precision score: {precision:.4f}")
print(f"LightGBM f1 score: {f1:.4f}")

svm_clf = SVC(kernel='rbf', random_state=0, gamma=0.5, C=1.0)
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)

accuracy = metrics.accuracy_score(y_pred, y_test)
recall = metrics.recall_score(y_pred, y_test)
precision = metrics.precision_score(y_pred, y_test)
f1 = metrics.f1_score(y_pred, y_test)

print(f"SVM accuracy score: {accuracy:.4f}")
print(f"SVM recall score: {recall:.4f}")
print(f"SVM precision score: {precision:.4f}")
print(f"SVM f1 score: {f1:.4f}")

gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)
y_pred = gnb_clf.predict(X_test)

accuracy = metrics.accuracy_score(y_pred, y_test)
recall = metrics.recall_score(y_pred, y_test)
precision = metrics.precision_score(y_pred, y_test)
f1 = metrics.f1_score(y_pred, y_test)

print(f"NaiveBayes accuracy score: {accuracy:.4f}")
print(f"NaiveBayes recall score: {recall:.4f}")
print(f"NaiveBayes precision score: {precision:.4f}")
print(f"NaiveBayes f1 score: {f1:.4f}")

