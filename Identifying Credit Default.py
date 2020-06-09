import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import pandas_profiling as pp
from pandas_profiling import ProfileReport
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

df = pd.read_csv("C:\\Users\shavl\PycharmProjects\Identifying Credit Default/UCI_Credit_Card.csv", index_col=0, na_values='')

X = df.copy()
y = X.pop('default.payment.next.month')

TEST_SIZE = 0.2

print(df.info())

print(df.describe().transpose().round(2))

msno.matrix(X)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, gamma=0.7, subsample = 0.8, max_depth=6, min_child_weight=3, random_state=0)
xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)
recall = metrics.recall_score(y_pred, y_test)
print(f"XGBoost recall score: {recall:.4f}")

lgb_clf = LGBMClassifier()
lgb_clf.fit(X_train, y_train)
y_pred = lgb_clf.predict(X_test)
recall = metrics.recall_score(y_pred, y_test)
print(f"LightBoost recall score: {recall:.4f}")

svm_clf = SVC(kernel='rbf', random_state=0, gamma=0.5, C=1.0)
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
recall = metrics.recall_score(y_pred, y_test)
print(f"SVM recall score: {recall:.4f}")

gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)
y_pred = gnb_clf.predict(X_test)
recall = metrics.recall_score(y_pred, y_test)
print(f"NaiveBayes recall score: {recall:.4f}")
