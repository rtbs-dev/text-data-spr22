import pandas as pd
import numpy as np
import pickle
import yaml
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import classification_report

from my_functions import preprocess

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)


loss = params["LinearSVC"]["loss"]
penalty = params["LinearSVC"]["penalty"]

# print(loss)

# print(penalty)
# exit()
# Read in magic data
df = (
    pd.read_feather('../../../data/mtg.feather')
    .dropna(subset=['flavor_text', 'text'])
    .reset_index(drop=True)
)

# Preprocess text columns
clean_text = preprocess(df.text)
clean_flavor_text = preprocess(df.flavor_text)

# Concatenate the 2 text columns together
X = []
for i in range(len(clean_text)):
    text_concat = clean_text[i] + ". " + clean_flavor_text[i]
    X.append(text_concat)
    
y = MultiLabelBinarizer().fit_transform(df.color_identity)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)              

model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', OneVsRestClassifier(LinearSVC(
        loss = loss,
        class_weight="balanced", random_state=42)))])

#fit model with training data
model.fit(X_train, y_train)

# save
with open('multilabel.pkl','wb') as f:
    pickle.dump(model,f)

y_pred = model.predict(X_test)

# print(label_ranking_loss(y_test, y_pred))

ranking_loss = label_ranking_loss(y_test, y_pred)
# print(classification_report(y_test, y_pred, output_dict=True))
metrics = {'penalty':penalty, 'loss':loss, 'label_ranking_loss': ranking_loss}


with open("metrics.json", "w") as outfile:
    json.dump(metrics, outfile)
    
exit()
