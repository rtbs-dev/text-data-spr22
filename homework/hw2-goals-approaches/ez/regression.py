"""Codes for regressionion model"""

# packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import pickle

# Load the data
df = pd.read_feather("../../../data/mtg.feather")
df.head()
# y: edhrec_rank
# X: converted_mana_cost, power, toughness, rarity, keywords, color_identity, types

# Remove NAs in column edhrec_rank
df = df.dropna(subset = ["edhrec_rank"]).reset_index(drop=True)

# converted_mana_cost
# convert to int
df["converted_mana_cost"] = df["converted_mana_cost"].astype(int)

# power
# Replace NaN power with 0
df["power"] = df["power"].fillna(0).astype(int)

# toughness
# Replace NaN toughness with 0
df["toughness"] = df["toughness"].fillna(0).astype(int)

# Numerical features
num_features = ["converted_mana_cost", "power", "toughness"]
num_transformer = Pipeline(
    steps=[("scaler", MinMaxScaler())]
)

# Categorical features
cat_features = ["rarity"]
cat_transformer = OneHotEncoder(handle_unknown="ignore")

# keywords
# no_keywords = 0 & yes_keywords = 1
df["keywords"] = df["keywords"].isnull().astype(int)

# color_identity
# Create a df for color_identity dummy variables
colors = pd.get_dummies(df.color_identity.apply(pd.Series).stack()).groupby(level=0).sum()
# Merge df with colors on index
df = df.join(colors)
# Replace NaN color_identity dummy variables with 0
df[colors.columns] = df[colors.columns].fillna(0).astype(int)

# types
# Create a df for types dummy variables
types = pd.get_dummies(df.types.apply(pd.Series).stack()).groupby(level=0).sum()
# Merge df with colors on index
df = df.join(types)
# Replace NaN types dummy variables with 0
df[types.columns] = df[types.columns].fillna(0).astype(int)

df = df[['converted_mana_cost', 'power', 'toughness', 'rarity',
         'keywords', 'B', 'G', 'R', 'U', 'W',
         'Artifact', 'Creature', 'Enchantment', 'Instant',
         'Land', 'Planeswalker', 'Sorcery', 'Tribal',
         'edhrec_rank']]


X = df.drop('edhrec_rank', axis=1)
y = df["edhrec_rank"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=111)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ],
    remainder='passthrough'
)

pipe = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("model", None)])

search_space = [

    # LinearRegression
    {'model' : [LinearRegression()]},

    # Lasso
    {'model' : [Lasso()]},

    # BaggingRegressor
    {'model' : [BaggingRegressor()]}
]

fold_generator = KFold(n_splits=15, shuffle=True,random_state=111)
search = GridSearchCV(
    pipe,
    search_space,
    cv=fold_generator,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2)

search.fit(X_train, y_train)
best = search.best_estimator_

best.score(X_test, y_test)

# Save the models using pickle
pickle.dump(best, open("regression.sav", 'wb'))
