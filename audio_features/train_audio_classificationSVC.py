import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from model import *
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

columns = [
    'zAggressive', 'zAttractive', 'zConfident', 'zMasculine',
    'zWin', 'zQuality', 'zIntel', 'zTrust']
mfcc_audio = pd.read_csv("data/output/mfcc_with_ratings.csv")
mfcc_audio.drop(['Unnamed: 0'], axis=1, inplace=True)

for ratingColumn in columns:
    mfcc_audio[ratingColumn][mfcc_audio[ratingColumn] < 0]=-1
    mfcc_audio[ratingColumn][mfcc_audio[ratingColumn] >= 0]=1

audios = np.unique(mfcc_audio["Audio"])
train_audio, test_audio = train_test_split(
    audios, train_size=0.7, test_size=0.3, random_state=0)

X_train = mfcc_audio[mfcc_audio["Audio"].isin(train_audio)]
X_test = mfcc_audio[mfcc_audio["Audio"].isin(test_audio)]
y_train = X_train[columns]
y_test = X_test[columns]

X_train.drop(columns + ["Audio"], inplace=True, axis=1)
X_test.drop(columns + ["Audio"], inplace=True, axis=1)

mor = MultiOutputClassifier(SVC(kernel='linear'), n_jobs=1)
mor.fit(X_train, y_train)

dummy = DummyClassifier(strategy='most_frequent',random_state=0)
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

estimators = mor.estimators_

for i, col in enumerate(columns):

    true = y_test[col]
    pred = estimators[i].predict(X_test)

    d_p = dummy_pred[:, i]

    print("accuracy score with SVC(kernel = rbf) " + col)
    score = accuracy_score(true, pred)
    print(score)

    print("dummy accuracy score " + col)
    score = accuracy_score(true, d_p)
    print(score)
