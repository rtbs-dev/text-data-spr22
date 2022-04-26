#!/usr/bin/env python
# coding: utf-8

# In[36]:


import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso

#Read in data
mtg = (pd.read_feather('../../../data/mtg.feather'))
#Drop rows with missing values in the variables of interest
mtg = mtg.dropna(subset = ['flavor_text', 'text', 'color_identity', 'edhrec_rank', 'rarity']).reset_index(drop=True)

#Create dummy columns based on color labels
for letter in np.unique(np.concatenate(np.array(mtg['color_identity']))):
    mtg["is_"+letter] = [1 if letter in x else 0 for x in mtg['color_identity']]
    
#Feature variables: text, flavor text, color labels, rarity
mtg = pd.get_dummies(mtg, columns = ['rarity'])

#Specify X and Y
y = mtg['edhrec_rank']

X = mtg[['text','flavor_text','is_B','is_G', 'is_R', 'is_U', 'is_W', 'rarity_uncommon', 'rarity_rare']]

#70/30 training-test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

#(1) Choose a number of folds and specify a random state
fold_generator = KFold(n_splits=5, shuffle=True)

#(2) Specify a preprocessing step for the text columns
preprocess = ColumnTransformer([('text', TfidfVectorizer(), 'text'),
                                ('flavor_text', TfidfVectorizer(), 'flavor_text')])

#(3) Create the model pipe
pipe = Pipeline(steps=[('pre_process', preprocess),
                       ('model',None)])

#(4) Instantiate the search space
search_space = [
    # Naive Bayes Classifier
    {'model' : [Lasso()]},
    
    # K-Nearest-Neighbors, also specifying values of K to test
    {'model' : [KNeighborsRegressor()],
    'model__n_neighbors':[5,10,25,50]},
    
    # Decision Tree, also specifying depth levels to test
    {'model': [DecisionTreeRegressor()],
    'model__max_depth':[2,3,4]},
    
    # Random forest, also specifying depth levels, numbers of estimators, and numbers of features to test
    {'model' : [RandomForestRegressor()],
    'model__max_depth':[2,3,4],
    'model__n_estimators':[500,1000,1500],
    'model__max_features':[3,4,5]},
]

#Assemble the GridSearch
search = GridSearchCV(pipe, 
                      search_space, 
                      cv = fold_generator,
                      scoring='neg_mean_squared_error',
                      n_jobs=-1)

#Fit the GridSearch
search.fit(X_train, y_train)

#Save the best estimator 
joblib.dump(search.best_estimator_, 'regression_GridSearch.pkl')

