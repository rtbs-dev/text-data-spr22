#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

#Read in data
mtg = (pd.read_feather('../../../data/mtg.feather'))
#Drop rows with missing values in the variables of interest
mtg = mtg.dropna(subset = ['flavor_text', 'text', 'color_identity']).reset_index(drop=True)

for letter in np.unique(np.concatenate(np.array(mtg['color_identity']))):
    mtg["is_"+letter] = [1 if letter in x else 0 for x in mtg['color_identity']]

letters = mtg.columns.str.contains('is_')

mtg['labels'] = mtg[mtg.columns[letters]].values.tolist()

#Select labels as targets
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(mtg['labels'])
#Select text columns as features
X = mtg[['text', 'flavor_text']]

#Training test split 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#Preprocess text (this took several hours to debug and I am honestly not joking)
preprocess = ColumnTransformer(transformers=[('text', TfidfVectorizer(), 'text'),
                                             ('flavor_text', TfidfVectorizer(), 'flavor_text')])

#Create pipeline with preprocessing and linear SVC
pipe = Pipeline([
    ('preprocess', preprocess),
    ('classifier', OneVsRestClassifier(SVC()))
])

#Fit pipe to training data
fitted_pipe = pipe.fit(X_train, y_train)

#Export pickeled pipe
joblib.dump(fitted_pipe, 'multilabel_pipe.pkl')

