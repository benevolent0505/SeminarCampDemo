#!/usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score


# データセット用意
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                 header=None)

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

le = LabelEncoder()
le.fit(np.unique(y))
y = le.transform(y)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)


# パイプライン作成
pipe_svm = Pipeline([('scl', StandardScaler()),
                    ('clf', SVC(random_state=1))])


scores = cross_val_score(estimator=pipe_svm,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=-1)

print('CV accuracy scores: {}'.format(scores))
print('CV accuracy: {:0.3f} +/- {:0.3f}'.format(np.mean(scores), np.std(scores)))

pipe_svm.fit(X_train, y_train)
print('Test accuracy: {:0.3f}'.format(pipe_svm.score(X_test, y_test)))
