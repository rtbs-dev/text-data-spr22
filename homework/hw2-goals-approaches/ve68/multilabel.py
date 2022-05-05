# ## Part 2 Supervised Classification - Multilabel
#
# Using only the `text` and `flavor_text` data, predict the color identity of cards: 
#
# Follow the sklearn documentation covered in class on text data and Pipelines to create a classifier that predicts which of the colors a card is identified as. 
# You will need to preprocess the target _`color_identity`_ labels depending on the task: 
#
# - Source code for pipelines
#     - in `multilabel.py`, do the same, but with a multilabel model (e.g. [here](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_multilabel.html#sphx-glr-auto-examples-miscellaneous-plot-multilabel-py)). You should now use the original `color_identity` data as-is, with special attention to the multi-color cards. 
# - in `dvc.yaml`, add these as stages to take the data and scripts as input, with the trained/saved models as output. 

# #### Reference
# https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# +
import yaml

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)
    
param_min_df = params['tfidf']['min_df']
param_ngram_range = tuple(map(int, params['tfidf']['ngram_range'].split(', ')))


# +
def multilabel(seed=2022):
    """
    Prepares MTG data (X, y) and exports multilabel model using LinearSVC
    :param seed: Seed to guarantee consistent results
    """
    import pandas as pd
    
    from sklearn.model_selection import train_test_split
    
    from sklearn import preprocessing
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import MultiLabelBinarizer
    
    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier    
    
    from sklearn import pipeline
    
#     from sklearn.pipeline import Pipeline
#     from sklearn.pipeline import FeatureUnion

    import pickle

    df = pd.read_feather("../../../data/mtg.feather")

    X = df['text'] + df['flavor_text'].fillna('')
    
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['color_identity'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    
    tfidf = TfidfVectorizer(
        min_df=param_min_df, 
        tokenizer=LemmaTokenizer(),
        ngram_range=param_ngram_range,
        stop_words='english')

    multilabel_model = OneVsRestClassifier(SVC(kernel='linear'))
    
    pipe = pipeline.make_pipeline(tfidf, multilabel_model)
    
    pipe.fit(X_train, y_train)
    
    pickle.dump(pipe, open('multilabel_pipe.sav', 'wb'))
    
#     multilabel_model_proba = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    
#     multilabel_model_proba.fit(X_train, y_train)
    
#     pickle.dump(multilabel_model_proba, open('multilabel_proba.sav', 'wb'))


# -

if __name__ == "__main__":
    multilabel()


