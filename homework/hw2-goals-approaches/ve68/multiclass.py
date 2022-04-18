# ## Part 2 Supervised Classification
#
# Using only the `text` and `flavor_text` data, predict the color identity of cards: 
#
# Follow the sklearn documentation covered in class on text data and Pipelines to create a classifier that predicts which of the colors a card is identified as. 
# You will need to preprocess the target _`color_identity`_ labels depending on the task: 
#
# - Source code for pipelines
#     - in `multiclass.py`, again load data and train a Pipeline that preprocesses the data and trains a multiclass classifier (`LinearSVC`), and saves the model pickel output once trained. target labels with more than one color should be _unlabeled_! 

def multiclass():
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split

    from sklearn.svm import LinearSVC

    from joblib import dump

    df = pd.read_feather("../../../data/mtg.feather")

    text = df['text'] + df['flavor_text'].fillna('')

    tfidf = TfidfVectorizer(
        min_df=5, 
        stop_words='english')

    X_tfidf = tfidf.fit_transform(text)
    
    ci = df['color_identity']
    
    single_color_identity = [list(i)[0] if len(i) == 1 else 0 for i in ci]
    
    le = preprocessing.LabelEncoder()
    
    y = le.fit_transform(single_color_identity)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, random_state = 20220418)
    
    model = LinearSVC()
    model.fit(X_train, y_train)
    dump(model, 'multiclass.joblib') 

if __name__ == "__main__":
    multiclass()


