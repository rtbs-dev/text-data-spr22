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
from bertopic import BERTopic
from sklearn.model_selection import train_test_split

df = (pd.read_feather('C:/Users/JeffW/text-data-spr22/data/mtg.feather', 
                      columns = ['name','text', 'mana_cost','flavor_text', 'release_date']
                     )
                     ).dropna(subset=['mana_cost', 'text', 'flavor_text'])

df['card_num'] = df.reset_index().index

# +
df = df.explode('mana_cost')

for i in range(10):
    df = df.loc[df["mana_cost"] != str(i)]

df.mana_cost.value_counts()
# -

count = df.groupby(['card_num'])['mana_cost'].count()
count = pd.DataFrame(count)
count

df = pd.merge(df, count, how='left', on=['card_num'], validate = 'm:1')

df = df[(df['mana_cost_y'] == 1)]
df= df[['name', 'text', 'mana_cost_x', 'flavor_text', 'release_date']]

df.rename(columns={'mana_cost_x': 'color_identity'}, inplace=True)
df['all_text'] = df['text'] + " " + df['flavor_text']
df

X = df['flavor_text']
y = df['color_identity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

docs = X_train

# +
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

X, y = make_classification(n_features=4, random_state = 0)
clf = make_pipeline(CountVectorizer(ngram_range=(1,3)),
                    TfidfTransformer(),
                    LinearSVC(random_state=0, tol=1e-05))
clf.fit(docs, y_train)
# -

import numpy as np
print(clf.named_steps['linearsvc'].coef_)
predicted = clf.predict(X_test)
np.mean(predicted == y_test)

# +
import pickle

with open("multiclass_model.pkl", 'wb') as file:
    pickle.dump(clf, file)
