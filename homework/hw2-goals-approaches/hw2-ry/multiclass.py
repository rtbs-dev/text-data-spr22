# import dependencies
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn import preprocessing

# data preparation
color = (pd.read_feather('../../../data/mtg.feather')# <-- will need to change for your notebook location
[['color_identity','flavor_text','text']]
)
## remove na
color = color[color['flavor_text'].notna()]
## remove multiple color identity cards as unlabel
color = color[color['color_identity'].map(len) < 2]

## X, y datasets
### X text
text = color.text.str.cat(others=[color.text, color.flavor_text], sep='\n').fillna('')
### y
y = []
for i in color['color_identity']:
    if len(i)==1:
        y.append(i[0])
    else:
        y.append(0)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

## X_train
tfidf = TfidfVectorizer(
        min_df=0.2,
        max_df = 0.8,
        stop_words='english')

X = tfidf.fit_transform(text)

# initialize model
multiclass_model = LinearSVC()
## seperate data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)

## fit multiclass_model
multiclass_model.fit(X_train, y_train)

# save model using pickle

pipe = pipeline.make_pipeline(tfidf, multilabel_model)

    pipe.fit(X_train, y_train)

    pickle.dump(pipe, open('multiclass_pipe.sav', 'wb'))

filename = 'multiclass_model.sav'
pickle.dump(multiclass_model, open(filename, 'wb'))
