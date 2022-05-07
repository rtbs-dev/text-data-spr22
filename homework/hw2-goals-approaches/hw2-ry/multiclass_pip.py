# import dependencies
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.metrics import classification_report
import yaml
import re


with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)
dfpercent = params["preprocessing"]["df"]
# data preparation

## tokenization
tokenize = re.compile(
    r"(?:\#[\w\d]+\b)"
    r"|(?:\b\w[\/\&]\w)\b"
    r"|(?:\b\w[\w\'\d]+)\b"
    r"|(?:\{\w\})"  # mana
    r"|(?:[+-]\d\d?(?:/[+-]\d\d?)?)"  # tokens
)


color = (
    pd.read_feather("../../../data/mtg.feather")
    .drop_duplicates(  # <-- will need to change for your notebook locatio
        subset=["name"]
    )
    .assign(
        color_singles=lambda df: df["color_identity"]
        .where(df["color_identity"].str.len() == 1, "")
        .str[0]
    )[["text", "flavor_text", "color_singles"]]
    .dropna()
)
## vectorization
vectorizer = TfidfVectorizer(
    min_df=0.2,
    max_df=0.8,
    stop_words="english",
    tokenizer=tokenize.findall,
)

multiclf = LinearSVC()
## pipeline
pipe = Pipeline(
    [
        ("vec", vectorizer),
        ("clf", multiclf),
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    color["text"].str.cat(color["flavor_text"], sep=" \n "),
    color["color_singles"],
    random_state=42,
)

pipe

pipe.fit(X_train, y_train)


y_pred = pipe.predict(X_test)

metrics = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
print(metrics)
metrics["weighted avg"].to_json("metrics.json")
pickle.dump(pipe, open('multiclass_pipe.sav', 'wb'))
