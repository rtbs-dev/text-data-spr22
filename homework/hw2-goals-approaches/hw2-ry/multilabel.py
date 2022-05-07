
# import dependencies
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

# data preparation
color = (pd.read_feather('../../../data/mtg.feather')# <-- will need to change for your notebook location
[['color_identity','flavor_text','text']]
)

## X, y datasets
text = color.text.str.cat(others=[color.text, color.flavor_text], sep='\n').fillna('')
y = color[['color_identity']]


### X_train
tfidf = TfidfVectorizer(
        min_df=0.2,
        max_df = 0.8,
        stop_words='english')

X = tfidf.fit_transform(text)

### y datasets
cv = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
y = cv.fit_transform(color['color_identity'])



# model preparation
multilabel_model = OneVsRestClassifier(SVC(kernel="linear"))
## data preparation
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)

#fit model
multilabel_model.fit(X_train, y_train)

#save model
filename = 'multilabel_model.sav'
pickle.dump(multilabel_model, open(filename, 'wb'))
