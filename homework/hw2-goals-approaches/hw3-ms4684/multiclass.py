#!/usr/bin/env python
# coding: utf-8

# Using only the text and flavor_text data, predict the color identity of cards:
# 
# Follow the sklearn documentation covered in class on text data and Pipelines to create a classifier that predicts which of the colors a card is identified as. You will need to preprocess the target color_identity labels depending on the task:
# 
# - Source code for pipelines
#     - **in multiclass.py, again load data and train a Pipeline that preprocesses the data and trains a multiclass classifier (LinearSVC), and saves the model pickel output once trained. Target labels with more than one color should be unlabeled!**
# 

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
import yaml
from sklearn import metrics
import json


# In[2]:


with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

ngrams = params["preprocessing"]["ngrams"]


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


# remove rows where target (color_identity) or predictors (flavor_text and text) have missing values
df2 = df.dropna(how = 'any',
                subset = ['flavor_text'])

# check
df2.isnull().sum()


# #### For $x$, combine text and flavor text data

# In[7]:


df2['combined_text'] = df['text'] + ' ' + df['flavor_text']

# view
df2.head(2)


# #### For $y$, encode target variable (`color_identity`)
# 
# Target labels with more than one color should be unlabeled!
# 
# To "unlabel" data, I will replace the label with -1.<br>
# Where there are no values, I will replace the label to null
# 

# In[8]:


# store color_identity values as a list
color_identity_values = list(df2.color_identity.values)

# create empty list to store results
color_identity_multiclass = []

# iterate through list, and unlabel target labels with more than one color
for i in color_identity_values:
    if len(i) == 1:
        color_identity_multiclass.append(i[0])
    elif len(i) < 1:
        color_identity_multiclass.append(0) # storing missing values as 0
    else:
        color_identity_multiclass.append(-1) # unlabeling target labels with more than one color

# check length
len(color_identity_multiclass)


# In[9]:


# check target labels
set(color_identity_multiclass)


# In[10]:


### encode target labels (I will do this manually instead of using LabelEncoder())

# store empty list to append to later
encoded_target_multiclass = []

for i in color_identity_multiclass:
    if i == 'W':
        encoded_target_multiclass.append(1)
    elif i == 'U':
        encoded_target_multiclass.append(2)
    elif i == 'R':
        encoded_target_multiclass.append(3)
    elif i == 'G':
        encoded_target_multiclass.append(4)
    elif i == 'B':
        encoded_target_multiclass.append(5)
    elif i == -1:
        encoded_target_multiclass.append(i)
    else:
        encoded_target_multiclass.append(i)
        
# check length
len(encoded_target_multiclass)


# In[11]:


# check labels
set(encoded_target_multiclass)


# In[12]:


# add encoded labels to dataframe as a new column
df2['multiclass'] = encoded_target_multiclass

# view
df2.head(2)


# #### Split data into training and test sets

# In[13]:


# store target and predictor
y = df2[['multiclass']]
X = df2[['combined_text']]

# split data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y , test_size = .25, random_state = 123)


# In[14]:


# check training and test data shapes
print(train_X.shape[0]/df2.shape[0])
print(test_X.shape[0]/df2.shape[0])


# #### Training Data

# In[15]:


# store training data as a list
training_X = train_X.combined_text.tolist()

# check length
len(training_X)


# In[16]:


# check train_y length
len(train_y)


# In[17]:


# store training target as numpy array
training_target = train_y.multiclass.values

# check length
len(training_target)


# #### Test Data

# In[18]:


# store test data as a list
test_x = test_X.combined_text.tolist()

# check length
len(test_x)


# In[19]:


# check test_y length
len(test_y)


# In[20]:


# store test target as numpy array
test_target = test_y.multiclass.values

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

# In[21]:


# create pipeline
multiclass_text_clf = Pipeline([('vect', CountVectorizer(stop_words = "english",
                                              lowercase = True,
                                              ngram_range = (ngrams["smallest"], ngrams["largest"]), # lower bound,upper bound: 1,2 unigrams and bigrams
                                              min_df = 5, # ignore rare words (appear in less than 5 documents)
                                              max_df = 0.9)), # ignore common words (appear in more than 90% of documents)
                     ('tfidf', TfidfTransformer()), 
                     ('clf', LinearSVC()),]) # Linear SVC

# train the model
multiclass_text_clf.fit(training_X, training_target)


# In[22]:


# store model
file_to_store = open("multiclass_classifier.pickle", "wb")
pickle.dump(multiclass_text_clf, file_to_store)
file_to_store.close()


# In[23]:


y_pred = multiclass_text_clf.predict(test_x)

metrics = metrics.classification_report(test_target, y_pred, output_dict=True)
print(metrics)
#metrics["weighted avg"].to_json("metrics.json")


# In[24]:


# print(label_ranking_loss(y_test, y_pred))
with open("metrics.json", "w") as outfile:
    json.dump(metrics, outfile)
    
exit()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




