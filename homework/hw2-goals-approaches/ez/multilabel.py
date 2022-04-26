"""Codes for multilabel model"""

# packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# Load the data
df = pd.read_feather("../../../data/mtg.feather")

# Remove NAs
df = df.dropna(subset = ["flavor_text", "text", "color_identity"]).reset_index(drop=True)

# X
tfidf = TfidfVectorizer(
    min_df=5, # ignore rare words (appear in less than 5 documents)
    max_df=0.8, # ignore common words (appear in more than 80% of documents)
    stop_words="english"
)
text = df["text"] + df["flavor_text"].fillna('')
X = tfidf.fit_transform(text)

# y
ci = df["color_identity"]
cv = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
y = cv.fit_transform(ci)

# Instantiate OneVsRestClassifier
multilabel_model = OneVsRestClassifier(SVC(kernel="linear"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=111)

# Fit OneVsRestClassifier
multilabel_model.fit(X_train, y_train)

# Save the model using pickle
pickle.dump(multilabel_model, open("multilabel.sav", 'wb'))
