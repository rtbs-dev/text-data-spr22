import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml 

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

min_df = params["tfidf"]["min_df"]

# read the data
mtg = pd.read_feather('../../../data/mtg.feather')[["flavor_text", "text", "color_identity"]]

# mapping
color_mapping = {"['W']": 'tan', "['U']": 'blue', "['B']": 'purple', "['R']": 'red', "['G']": 'green', "['M']": 'goldenrod'}
mtg['color_identity'] = mtg['color_identity'].astype(str).map(color_mapping)

# dropna
mtg.dropna(inplace=True)

# join the text together
text = mtg.text.str.cat(mtg.flavor_text, sep='\n')
text

# tokenizer
import re
tokenize = re.compile(
    r'(?:\#[\w\d]+\b)'
    r'|(?:\b\w[\/\&]\w)\b'
    r'|(?:\b\w[\w\'\d]+)\b'
    r'|(?:\{\w\})'  # mana
    r'|(?:[+-]\d\d?(?:/[+-]\d\d?)?)'  # tokens
)

text.str.findall(tokenize).explode().unique()[:100]

# train a multiclass classifier for mtg in a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
    tokenizer=tokenize.findall,
    min_df=min_df, 
    max_df=0.8,
    stop_words='english',
    ngram_range=(1,2))),
    ('clf', LinearSVC())
])

# train test split
X_train, X_test, y_train, y_test = train_test_split(text, mtg["color_identity"], test_size=0.2, random_state=42)

# fit the pipeline
pipeline.fit(X_train, y_train)

# save model
import pickle   
# save the model with pickle- change this path
with open('mtg_classifier.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# score the pipeline
y_pred =  pipeline.predict(X_test)
report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
metrics = report.transpose()["f1-score"]

# print metrtics to json
metrics.to_json("metrics.json")
