# import dependencies
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing

#import os
#os.getcwd()
#os.chdir('text-data-spr22/homework/hw2-goals-approaches/hw-mk')

def multiclass_model():
    '''
    Runs and saves mutliclass (LinearSVC) model on magic the gathering flavor text data.
    '''
    # Data Prep
    df = pd.read_feather('../../../data/mtg.feather')
    df = df[df['flavor_text'].notna()]
    ## remove multiple color identity cards as unlabel
    df = df[df['color_identity'].map(len) < 2]

    # Encoders
    ### X
    text = df.text.str.cat(others=[df.text, df.flavor_text], sep='\n').fillna('')
    ### y
    y = []
    for i in df['color_identity']:
        #print(i[0], i)
        if len(i)==1:
            y.append(i[0])
        else:
            y.append(0)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    # TFIDF Vectorize
    tfidf = TfidfVectorizer(
            min_df=0.2,
            max_df = 0.8,
            stop_words='english')

    X = tfidf.fit_transform(text)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)

    # LinearSVC Model
    multiclass_model = LinearSVC()
    multiclass_model.fit(X_train, y_train)
    pickle.dump(multiclass_model, open('multiclass_model.sav', 'wb'))

result = multiclass_model.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(multiclass_model, X_test, y_test)

plt.savefig('foo.png')

if __name__ == "__main__":
    multilabel_model()
