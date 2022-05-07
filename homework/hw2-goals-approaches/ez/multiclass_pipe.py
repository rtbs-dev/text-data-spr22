"""Codes for multiclass model"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np
import yaml
import pandas as pd
import pickle

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

ngrams = params["preprocessing"]["ngrams"]

tokenize = re.compile(
    r"(?:\#[\w\d]+\b)"
    r"|(?:\b\w[\/\&]\w)\b"
    r"|(?:\b\w[\w\'\d]+)\b"
    r"|(?:\{\w\})"  # mana
    r"|(?:[+-]\d\d?(?:/[+-]\d\d?)?)"  # tokens
)

df = (
    pd.read_feather("../../../data/mtg.feather")
    .drop_duplicates(
        subset=["name"]
    )
    .assign(
        color_singles=lambda df: df["color_identity"]
        .where(df["color_identity"].str.len() == 1, "")
        .str[0]
    )[["text", "flavor_text", "color_singles"]]
    .dropna()
)

vectorizer = TfidfVectorizer(
    min_df=5,
    max_df=0.8,
    stop_words="english",
    ngram_range=(ngrams["smallest"], ngrams["largest"]),
    tokenizer=tokenize.findall,
)

multiclf = LinearSVC()

pipe = Pipeline(
    [
        ("vec", vectorizer),
        ("clf", multiclf),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"].str.cat(df["flavor_text"], sep=" \n "),
    df["color_singles"],
    random_state=111,
)

pipe.fit(X_train, y_train)

# Save the model using pickle
pickle.dump(pipe, open("multiclass_pipe.sav", 'wb'))

y_pred = pipe.predict(X_test)

metrics = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
metrics["weighted avg"].to_json("metrics.json")
