#depedencies

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
from sklearn.linear_model import Lars
from tqdm.notebook import tqdm as notebook_tqdm
from bertopic import BERTopic
from sklearn.linear_model import LinearRegression
import numpy as np

'''
# this is my initial try, did not work well
df = df[df['edhrec_rank'].notna()]
df=df[['edhrec_rank','text','flavor_text']]



## X, y datasets
text = df.text.str.cat(others=[df.text, df.flavor_text], sep='\n').fillna('')
y=df['edhrec_rank'].tolist()


### X_train
tfidf = TfidfVectorizer(
        min_df=0.2,
        max_df = 0.8,
        stop_words='english')

X = tfidf.fit_transform(text)

#fit model
#lars
reg_model = Lars( normalize=False)

## seperate data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)

## fit multiclass_model
X_train = X_train.toarray()
reg_model.fit(X_train, y_train)

# save model using pickle

filename = 'reg_model.sav'
pickle.dump(reg_model, open(filename, 'wb'))
'''

## topic modeling combined
## X, y datasets
df = (pd.read_feather('../../../data/mtg.feather'))
df = df[df['flavor_text'].notna()]
#model training-topic modelling classification
topic_model = BERTopic()
docs = df['flavor_text'].dropna().tolist()
topics, probs = topic_model.fit_transform(docs)
new_model = topic_model.reduce_topics(docs, topics, nr_topics=20)
topic_list = new_model[0]
df['topic_rank']= topic_list

#prepare data for regression modeling
X =df[['topic_rank','''power','converted_mana_cost','toughness''']]
y =df['edhrec_rank']
X = X.replace(np.nan, 0)
y = y.replace(np.nan, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)

## Lars model
reg_model = Lars( normalize=False)
reg_model.fit(X_train,y_train)

#save model
# save model using pickle

filename = 'rank_model.sav'
pickle.dump(reg_model, open(filename, 'wb'))
