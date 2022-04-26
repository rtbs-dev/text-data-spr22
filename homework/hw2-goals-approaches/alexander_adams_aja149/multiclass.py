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
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

#Read in data
mtg = (pd.read_feather('../../../data/mtg.feather'))
#Drop rows with missing values in the variables of interest
mtg = mtg.dropna(subset = ['flavor_text', 'text', 'color_identity']).reset_index(drop=True)

#Create numeric labels based on values of colors
#New values: X = multiclass, Z = NaN
mtg['color_label'] = [np.array(['X']) if len(x) > 1 else x for x in mtg['color_identity']]
mtg['color_label'] = [np.array(['Z']) if len(x) == 0 else x for x in mtg['color_label']]
mtg['color_label'] = np.concatenate(mtg['color_label'])

#Merge labels into MTG data frame
labels = pd.DataFrame(mtg['color_label'].unique()).reset_index()
#Add one because zero indexed
labels['index'] = labels['index']+1
labels.columns = ['label', 'color_label']
mtg = labels.merge(pd.DataFrame(mtg), how = 'right', on = 'color_label')

#Select labels as targets
y = mtg['label']

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
    ('LinearSVC', LinearSVC())
])

#Fit pipe to training data
fitted_pipe = pipe.fit(X_train, y_train)

#Export pickeled pipe
joblib.dump(fitted_pipe, 'multiclass_pipe.pkl')

