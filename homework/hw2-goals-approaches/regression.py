# Read in magic data
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import BaggingRegressor

from sklearn.model_selection import KFold # Cross validation 
from sklearn.model_selection import cross_validate # Cross validation 
from sklearn.model_selection import GridSearchCV # Cross validation + param. tuning.

from my_functions import multi

df = (
    pd.read_feather('../../data/mtg.feather')
    .dropna(subset = ['edhrec_rank'])
    .reset_index(drop=True)
)
# Source: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html


numeric_features = ["converted_mana_cost"]
numeric_transformer = Pipeline(
    steps=[("scaler", MinMaxScaler())]
)

cat_features = ["block","rarity"]
cat_transformer = OneHotEncoder(handle_unknown="ignore")
multi_label = multi(df,["types","subtypes", "color_identity","supertypes"])

X = pd.concat([df[['converted_mana_cost','rarity',"block"]],multi_label], axis = 1)
y = df['edhrec_rank']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-scikit-learn/
# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", cat_transformer, cat_features)
    ],
    remainder='passthrough'
)

pipe = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("model", None)])

search_space = [

    # KNN with K tuning param
    # {'model' : [KNeighborsRegressor()],
    #  'model__n_neighbors':[5,10,15,20,25]},
    
    # # Decision Tree with the Max Depth Param
    # {'model': [DecisionTreeRegressor()],
    #  'model__max_depth':[2,3,4]},
    
    # # Random forest with the N Estimators tuning param
    # {'model' : [RandomForestRegressor()],
    # 'model__max_depth':[2,3,4],
    # 'model__n_estimators':[100,500,1000]},
    
    # # Ridge
    # {'model' : [Ridge()],
    # 'model__alpha':[0.25,0.5,0.75,1.0]},
        
    # # Lasso
    # {'model' : [Lasso()],
    # 'model__alpha':[0.25,0.5,0.75,1.0]},

    
    # # SVR
    # {'model' : [SVR()],
    # 'model__epsilon':[0.1,0.25,0.5,0.75]},
    
    # Bag
    {'model' : [BaggingRegressor()]}
]

fold_generator = KFold(n_splits=15, shuffle=True,random_state=42)

search = GridSearchCV(pipe, search_space, 
                      cv = fold_generator,
                      scoring='neg_mean_squared_error',
                      n_jobs=-1, 
                      verbose = 2)

search.fit(X_train,y_train)

best = search.best_estimator_

# save
with open('regression.pkl','wb') as f:
    pickle.dump(best,f)
    
exit()

