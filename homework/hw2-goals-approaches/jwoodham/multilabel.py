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
import pickle
import re
from bertopic import BERTopic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

df = (pd.read_feather('../../../data/mtg.feather', 
                      columns = ['name','text', 'color_identity','flavor_text', 'release_date']
                     )
                     ).dropna(subset=['name', 'color_identity', 'text', 'flavor_text', 'release_date'])

df['card_num'] = df.reset_index().index
df = df.reset_index(drop = True)
df
# -

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['color_identity'])

tokenize = re.compile(
    r"(?:\b\w[\w\'\d]+)\b"
)

X = df.flavor_text

X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

X, y = make_classification(n_features=4, random_state = 0)
clf = make_pipeline(CountVectorizer(stop_words = 'english', ngram_range=(1, 2), tokenizer = tokenize.findall),
                    TfidfTransformer(),
                    OneVsRestClassifier(SVC()))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
np.mean(y_pred == y_test)

# +
import pickle

with open("multilabel_model.pkl", 'wb') as file:
    pickle.dump(clf, file)
