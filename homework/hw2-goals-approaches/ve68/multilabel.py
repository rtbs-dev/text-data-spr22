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

def multilabel():
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import preprocessing
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import LinearSVC
    from sklearn.multiclass import OneVsRestClassifier

    import pickle

    df = pd.read_feather("../../../data/mtg.feather")

    text = df['text'] + df['flavor_text'].fillna('')

    # Convert this into scikit-learn pipeline?
    tfidf = TfidfVectorizer(
        min_df=5, 
        stop_words='english')

    X_tfidf = tfidf.fit_transform(text)

    ci = df['color_identity']

    cv = CountVectorizer(tokenizer=lambda x: x, lowercase=False)

    y = cv.fit_transform(ci)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, random_state = 20220418)
    
    multilabel_model = OneVsRestClassifier(LinearSVC(), n_jobs=1)
    
    multilabel_model.fit(X_train, y_train)
    
    pickle.dump(multilabel_model, open('multilabel.sav', 'wb'))

if __name__ == "__main__":
    multilabel()


