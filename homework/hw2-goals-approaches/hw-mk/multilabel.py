import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

import pickle

#import os
#os.getcwd()
#os.chdir('text-data-spr22/homework/hw2-goals-approaches/hw-mk')


def multilabel_model():
    """
    Runs LinearSVC on magic the gather flavor text.
    """

    df = pd.read_feather("../../../data/mtg.feather")

    text = df['text'] + df['flavor_text'].fillna('')

    tfidf = TfidfVectorizer(
        min_df=5,
        stop_words='english')

    X_tfidf = tfidf.fit_transform(text)
    ci = df['color_identity']

    cv = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
    y = cv.fit_transform(ci)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, random_state=1234)

    multilabel_model = OneVsRestClassifier(LinearSVC(), n_jobs=1)
    multilabel_model.fit(X_train, y_train)
    pickle.dump(multilabel_model, open('multilabel_model.sav', 'wb'))

if __name__ == "__main__":
    multilabel_model()
