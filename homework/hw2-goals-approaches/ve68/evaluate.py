# ## Part 4: Iteration, Measurement, & Validation
#
# Pick ONE of your models above (regression, multilabel, or multiclass) that you want to improve or investigate, and calculate metrics of interest for them to go beyond our confusion matrix/predicted-actual plots:
#
# * for multiclass, report average and F1
# * for multilabel, report an appropriate metric (e.g. `ranking_loss`)
# * for regression, report an appropriate metric (e.g. 'MAPE' or MSE), OR since these are ranks, the pearson correlation between predicted and actual may be more appropriate?
#
# in the corresponding `dvc.yaml` stage for your model-of-interest, add `params` and `metrics`
# * under `params`, pick a setting in your preprocessing (e.g. the TfidfVecorizer) that you want to change to imrpove your results. Set that param to your current setting, and have your model read from a `params.yaml` rather than directly writing it in your code.
# * under `metrics`, reference your `metrics.json` and have your code write the results as json to that file, rather than simply printing them or reporting them in the notebook.
# * commit your changes to your branch, `run dvc repro dvc.yaml` for your file, then run a new experiment that changes that one parameter: e.g. `dvc exp run -S preprocessing.ngrams.largest=1` (see the example/ folder for a complete working example).
#
# Report the improvement/reduction in performance with the parameter change for your metric, whether by copy-pasting or using !dvc exp diff in the notebook, the results of dvc exp diff.

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def evaluate(seed=2022):
    """
    Evaluate multilabel_pipe with test data and export metrics
    :param seed: Seed to guarantee consistent results
    """
    import pandas as pd
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer 
    from sklearn.metrics import label_ranking_loss
    
    import pickle
    import json

    df = pd.read_feather("../../../data/mtg.feather")

    X = df['text'] + df['flavor_text'].fillna('')
    
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['color_identity'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    
    multilabel_pipe = pickle.load(open('multilabel_pipe.sav', 'rb'))
    
    y_pred = multilabel_pipe.predict(X_test)
    
    ranking_loss = label_ranking_loss(y_test, y_pred)

    with open('metrics.json', "w") as fd:
        json.dump({"ranking_loss": ranking_loss}, fd, indent=4)

if __name__ == "__main__":
    evaluate()


