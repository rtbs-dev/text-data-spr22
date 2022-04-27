# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python [conda env:text-data-class]
#     language: python
#     name: conda-env-text-data-class-py
# ---

# +
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

df = (pd.read_feather('C:/Users/JeffW/text-data-spr22/data/mtg.feather', 
                      columns = ['name','text', 'mana_cost','flavor_text', 'release_date']
                     )
                     ).dropna(subset=['name', 'mana_cost', 'text', 'flavor_text', 'release_date'])

df['card_num'] = df.reset_index().index
df = df.reset_index(drop = True)
df
# -

mlb = MultiLabelBinarizer(classes = ['B','C','G','R','S','U','W','X'])
y = mlb.fit_transform(df['mana_cost'])
mlb.classes_

pd.DataFrame(y, columns = mlb.classes_)

tfidf = TfidfVectorizer(ngram_range=(1,2))
X = tfidf.fit_transform(df.flavor_text)

X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

clf = OneVsRestClassifier(svc)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
np.mean(y_pred == y_test)

clf.predict(X_test)

mlb.inverse_transform(clf.predict(X_test))

# +
import pickle

with open("multilabel_model.pkl", 'wb') as file:
    pickle.dump(clf, file)
