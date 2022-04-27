#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Dependencies
import pandas as pd
### Text pre-processing 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
### building a pipeline
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


# In[2]:


df = (pd.read_feather('C:/Users/VIOLIN/Desktop/text-data-spr22/data/mtg.feather')[['name','text', 'mana_cost', 'flavor_text','release_date', 'edhrec_rank','color_identity']])
df #loading data


# In[3]:


df.shape #data exploration


# In[4]:


df.dtypes #data exploration


# In[5]:


df.isnull().any() #data exploration


# In[6]:


df.isnull().sum()  #data exploration


# In[7]:


df = df.dropna() #data exploration


# In[8]:


df.isnull().any() #data exploration


# In[9]:


df.shape #data exploration


# In[10]:


df['full_text'] = df['text'] + df['flavor_text'] #predictor variable


# In[11]:


x = df['full_text'].fillna('') #handling nulls
x.isnull().any()


# In[12]:


y = MultiLabelBinarizer().fit_transform(df.color_identity) ##converting y into binaries


# In[13]:


##splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y)


# In[14]:


##Pipe line with text preprocessing, tfidf & model
text_clf = Pipeline([
('vect', CountVectorizer(input='content',ngram_range=(1,1),max_df=25, min_df=5)),
('tfidf', TfidfTransformer()),
('model', OneVsRestClassifier(LinearSVC(), n_jobs=1))])


# In[15]:


text_clf.fit(x_train, y_train) ##fitting the model on training data


# In[16]:


import pickle


# In[17]:


pickle.dump(text_clf, open('multilabel.sav', 'wb')) #exporting the model object


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




