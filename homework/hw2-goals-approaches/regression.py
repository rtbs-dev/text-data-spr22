import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso

import pickle

#import os
#os.getcwd()
#os.chdir('text-data-spr22/homework/hw2-goals-approaches/hw-mk')

def regression_models():
    """
    Runs ElasticNet and Lasso Regression on magic the gathering flavor text data.
    """
    # Data prep
    df = pd.read_feather("../../../data/mtg.feather")
    df = df[df['edhrec_rank'].notna()]
    df['text_flavor_text'] = df['text'] + df['flavor_text'].fillna('')

    tfidf = TfidfVectorizer(
        min_df=5,
        stop_words='english')

    X_tfidf = tfidf.fit_transform(df['text_flavor_text'])
    y = df['edhrec_rank']
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.20, random_state=1234)

    # ElasticNet Modeling
    elasticnet_model = ElasticNet(random_state=0)
    elasticnet_model.fit(X_train, y_train)
    pickle.dump(elasticnet_model, open('regression_elasticnet_model.sav', 'wb'))

    # Lasso Modeling
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)
    pickle.dump(lasso_model, open('regression_lasso_model.sav', 'wb'))


if __name__ == "__main__":
    regression_models()
