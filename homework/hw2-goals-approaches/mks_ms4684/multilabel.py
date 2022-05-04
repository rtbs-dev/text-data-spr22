#!/usr/bin/env python
# coding: utf-8

# Using only the text and flavor_text data, predict the color identity of cards:
# 
# Follow the sklearn documentation covered in class on text data and Pipelines to create a classifier that predicts which of the colors a card is identified as. You will need to preprocess the target color_identity labels depending on the task:
# 
# - Source code for pipelines
#     - **in multilabel.py, do the same, but with a multilabel model (e.g. here). You should now use the original color_identity data as-is, with special attention to the multi-color cards.**

# In[1]:


# import modules
import pandas as pd
import numpy as np
from bertopic import BERTopic
import re
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


# In[3]:


# load in data
(pd.read_feather('C:/Georgetown University/Courses/Spring Semester 2022/Text As Data/text-data-spr22/data/mtg.feather')# <-- will need to change for your notebook location
 .head(2)  
)


# In[4]:


# store full data
df = (pd.read_feather('C:/Georgetown University/Courses/Spring Semester 2022/Text As Data/text-data-spr22/data/mtg.feather')  
)

# check shape
df.shape


# In[5]:


# check missing values
df.isnull().sum()


# `color_identity` and `text` don't have any missing values so only missing values from the `flavor_text` variable need to be removed.

# In[6]:


# remove rows where target (color_identity) or training data (flavor_text and text) have missing values
df2 = df.dropna(how = 'any',
                subset = ['flavor_text'])

# check
df2.isnull().sum()


# #### For $x$, combine text and flavor text data

# In[7]:


df2['combined_text'] = df['text'] + ' ' + df['flavor_text']

# view
df2.head(2)


# #### For $y$, use the (`color_identity`) column as is
# 
# Guidance obtained from: https://scikit-learn.org/stable/modules/preprocessing_targets.html#preprocessing-targets

# In[41]:


# store color_identity values as a list
color_identity_values = list(df2.color_identity.values)

# create label binary indicator array - target
color_identity_multilabels = MultiLabelBinarizer().fit_transform(color_identity_values)


# In[45]:


# store target and predictor
y = color_identity_multilabels
X = df2[['combined_text']]

# split data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y , test_size = .25, random_state = 123)


# In[46]:


# check training and test data shapes
print(train_X.shape[0]/df2.shape[0])
print(test_X.shape[0]/df2.shape[0])


# #### Training Data

# In[47]:


# store training data as a list
training_X = train_X.combined_text.tolist()

# check length
len(training_X)


# In[48]:


# check train_y length
len(train_y)


# In[49]:


# store training target as numpy array
training_target = train_y

# check length
len(training_target)


# #### Test Data

# In[51]:


# store test data as a list
test_x = test_X.combined_text.tolist()

# check length
len(test_x)


# In[52]:


# check test_y length
len(test_y)


# In[53]:


# store test target as numpy array
test_target = test_y

# check length
len(test_target)


# #### Create Pipeline and train the model

# Pre-processing text using CountVectorizer() arguments:
# - removing English stop words in order to remove the 'low-level' information in the text and focus more on the important information.
# - converting all words to lowercase - assumption is that the meaning and significance of a lowercase word is the same as when that word is in uppercase or capitalized. This will help remove noise.
# - ngram_range set to 1,2 i.e. capturing both unigrams and bigrams since Magic Card texts often have names/terms that are bigrams e.g. Soul Warden and Beetleback Chief. 
# - min_df set to 5 i.e. rare words that appear in less than 5 documents will be ignored.
# - max_df set to 0.9 i.e. words that appear in more than 90% of the documents will be ignored since they are not adding much to a specific document.
# 
# Using TfidfTransformer():
# - Term frequencies calculated to overcome the discrepancies with using occurence count for differently sized documents. 
# - Downscaled weights for words that occur in many documents and therefore do not add a lot of information than those that occur in a smaller share of the corpus (tf-idf)
# 

# In[58]:


# create pipeline
multilabel_text_clf = Pipeline([('vect', CountVectorizer(stop_words = "english",
                                              lowercase = True,
                                              ngram_range = (1, 2), # lower bound,upper bound: 1,2 unigrams and bigrams
                                              min_df = 5, # ignore rare words (appear in less than 5 documents)
                                              max_df = 0.9)), # ignore common words (appear in more than 90% of documents)
                     ('tfidf', TfidfTransformer()), 
                     ('clf', OneVsRestClassifier(SVC(kernel="linear"))),]) # Linear SVC

# train the model
multilabel_text_clf.fit(training_X, training_target)


# In[59]:


# store model
file_to_store = open("multilabel_classifier.pickle", "wb")
pickle.dump(multilabel_text_clf, file_to_store)
file_to_store.close()


# In[ ]:




