import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from my_functions import preprocess

# Read in magic data
df = (
    pd.read_feather('../../data/mtg.feather')
    .dropna(subset=['flavor_text', 'text'])
    .reset_index(drop=True)
)

# Keep rows where the len of color_identity is 1
df = df[df['color_identity'].map(lambda d: len(d)) == 1].reset_index(drop=True)

# Because all lists in df.color_identity now has length 1, take the first item from each list
y = df.color_identity.apply(lambda x: x[0])

# Preprocess text columns
clean_text = preprocess(df.text)
clean_flavor_text = preprocess(df.flavor_text)

# Concatenate the 2 text columns together
X = []
for i in range(len(clean_text)):
    text_concat = clean_text[i] + ". " + clean_flavor_text[i]
    X.append(text_concat)


# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)              

model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced", random_state=42)))])

#fit model with training data
model.fit(X_train, y_train)

# save
with open('multiclass.pkl','wb') as f:
    pickle.dump(model,f)
    
exit()
