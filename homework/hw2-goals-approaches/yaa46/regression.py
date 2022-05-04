
# import packages
import pandas as pd
from sklearn.linear_model import LinearRegression
import re
from sklearn.model_selection import train_test_split
import pickle  
from sklearn.feature_extraction.text import TfidfVectorizer

# first selection based just off what sounds importnat
mtg = pd.read_feather('../../../data/mtg.feather')[['edhrec_rank', 'color_identity', 'converted_mana_cost', 'power', 'toughness', 'rarity', 'subtypes', \
'supertypes', 'types', 'text', 'flavor_text', 'life', 'block']]

# what columns do we have
mtg.columns

# how many na's do we have
mtg.isna().sum()

# lots of nas in power, toughness, and life

# drop power, toughness, and life
mtg.drop(['power', 'toughness', 'life'], axis=1, inplace=True)

# also a lot of missingness in flavor text and block but let's run it with them first

# there are so many different possibilities for supertypes and subtypes let's drop them as well
mtg.drop(['supertypes', 'subtypes'], axis=1, inplace=True)

# dummy variables for rarity
mtg = pd.get_dummies(mtg, columns=['rarity'], drop_first=True)

# dummies for blocks (what about na's?)
mtg = pd.get_dummies(mtg, columns=['block'], drop_first=True)

# there are 33 types let's treat them as categorical and make them dummies too

# make string first
mtg.types = mtg.types.astype(str)

# now dummy
mtg = pd.get_dummies(mtg, columns=['types'], drop_first=True)

# 32 different possible color identity combos- let's make them dummies
mtg["color_identity"] = mtg["color_identity"].astype(str)
mtg = pd.get_dummies(mtg, columns=['color_identity'], drop_first=True)

# combine text and flavor text- space character prevents nas for missing flavor text
mtg.text = mtg.text.str.cat(mtg.flavor_text, sep='\n', na_rep='')

# drop flavor text
mtg.drop(['flavor_text'], axis=1, inplace=True)

# drop nas
mtg.dropna(inplace=True)

# separate text
text = mtg.text
mtg.drop(['text'], axis=1, inplace=True)

# build tokenizer
tokenize = re.compile(
    r'(?:\#[\w\d]+\b)'
    r'|(?:\b\w[\/\&]\w)\b'
    r'|(?:\b\w[\w\'\d]+)\b'
    r'|(?:\{\w\})'  # mana
    r'|(?:[+-]\d\d?(?:/[+-]\d\d?)?)')  # tokens

# tfidf
vect = TfidfVectorizer(
    tokenizer=tokenize.findall,
    min_df=10, 
    max_df=0.8,
    stop_words='english',
    ngram_range=(1,2))


# fit and transform and make dtm
X =vect.fit_transform(text)
term_indices = {index: term for term, index in vect.vocabulary_.items()}
colterms = [term_indices[i] for i in range(X.shape[1])]
dtm = pd.DataFrame(X.toarray(), columns=colterms)

# combine dtm and mtg
mtg = pd.concat([mtg, dtm], axis=1)
mtg.dropna(inplace=True)

# target and features w/o text
y = mtg.edhrec_rank 
X = mtg.drop(['edhrec_rank'], axis=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train linear regression
clf = LinearRegression().fit(X_train, y_train)

# test clf
clf.score(X_test, y_test)

# regression using text
# train test split
# X_train, X_test, y_train, y_test = train_test_split(dtm, y, test_size=0.2, random_state=42)

# train logistic regression
# clf = LinearRegression().fit(X_train, y_train)

# test clf
# clf.score(X_test, y_test)

# save the model with pickle
with open('regression.pkl', 'wb') as f:
     pickle.dump(clf, f)