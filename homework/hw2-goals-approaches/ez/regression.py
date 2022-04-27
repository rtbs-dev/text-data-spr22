"""Codes for regressionion model"""

# packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
import pickle

# Load the data
df = pd.read_feather("../../../data/mtg.feather")

# y: edhrec_rank
# X: converted_mana_cost, power, toughness, color_identity, keywords, rarity

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

# color_identity
# Create a df for color_identity dummy variables
colors = pd.get_dummies(df.color_identity.apply(pd.Series).stack()).sum(level=0)
# Merge df with colors on index
df = df.join(colors)
# Replace NaN color_identity dummy variables with 0
df[colors.columns] = df[colors.columns].fillna(0).astype(int)

# keywords
# no_keywords = 0 & yes_keywords = 1
df["keywords"] = df["keywords"].isnull().astype(int)

# rarity
# Create a dummy variable for each rarity level
df = pd.get_dummies(df, columns=["rarity"])


df = df[['converted_mana_cost', 'power', 'toughness',
         'B', 'G', 'R', 'U', 'W', 'keywords',
         'rarity_rare', 'rarity_common', 'rarity_uncommon',
         'edhrec_rank']]

y = df['edhrec_rank']
X = df.drop('edhrec_rank', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=111)

# Fit LinearRegression
linear = LinearRegression()
linear.fit(X_train, y_train)

# Fit Lasso
lasso = Lasso()
lasso.fit(X_train, y_train)

# Save the models using pickle
pickle.dump(linear, open("linear.sav", 'wb'))
pickle.dump(linear, open("lasso.sav", 'wb'))
