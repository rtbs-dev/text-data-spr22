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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from nltk.tokenize import RegexpTokenizer

df = (pd.read_feather('C:/Users/JeffW/text-data-spr22/data/mtg.feather',
                      columns = ['color_identity', 'converted_mana_cost', 'edhrec_rank', 'power', 'rarity', 'toughness', 'flavor_text']
     )).dropna()
df

df['flavor_text'] = df['flavor_text'].str.replace(r"[()-.,!?@\'\`\"\_\n]", " ")
df['flavor_text'] = df['flavor_text'].str.lower()
tokenizer = RegexpTokenizer(r'\w+')
df['flavor'] = df['flavor_text'].apply(tokenizer.tokenize)
df['flavor'] = df['flavor_text'].str.split(',').str.join(' ')
df

X = df['flavor']
y = df['edhrec_rank']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# +
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
# -

clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train_tfidf, y)

# +
import pickle

with open("regression.pkl", 'wb') as file:
    pickle.dump(clf, file)
