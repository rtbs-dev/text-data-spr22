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

import re
import pandas as pd
import numpy as np
import yaml
from bertopic import BERTopic
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# +
with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

ngrams = params["preprocessing"]["ngrams"]

# +
df = (pd.read_feather('../../../data/mtg.feather', 
                      columns = ['name','text', 'color_identity','flavor_text', 'release_date']
                     )
                     ).dropna(subset=['color_identity', 'text', 'flavor_text'])

df['card_num'] = df.reset_index().index
# -

df.color_identity.where(df.color_identity.str.len() == 1, np.nan, inplace=True)

df = df.dropna(subset=['color_identity'])

X = df['flavor_text']
y = df['color_identity']
y = y.apply(lambda x: x[0])

tokenize = re.compile(
    r"|(?:\b\w[\w\'\d]+)\b"
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# +
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

X, y = make_classification(n_features=4, random_state = 0)
clf = make_pipeline(CountVectorizer(stop_words = 'english', ngram_range=(ngrams["min"], ngrams["max"]), tokenizer = tokenize.findall),
                    TfidfTransformer(),
                    LinearSVC(random_state=0, tol=1e-05))
# -

clf.fit(X_train, y_train)

import numpy as np
print(clf.named_steps['linearsvc'].coef_)
predicted = clf.predict(X_test)
np.mean(predicted == y_test)

# +
import pickle

with open("multiclass_model.pkl", 'wb') as file:
    pickle.dump(clf, file)
# -

y_pred = clf.predict(X_test)

metrics = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
print(metrics)
metrics["weighted avg"].to_json("metrics.json")
