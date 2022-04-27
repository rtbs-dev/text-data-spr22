#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Dependencies
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
# Import label encoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle
#building a pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# In[2]:


##Loading data
df = (pd.read_feather('C:/Users/VIOLIN/Desktop/text-data-spr22/data/mtg.feather')[['name','text', 'mana_cost', 'flavor_text','release_date', 'edhrec_rank','color_identity']])
df


# In[3]:


df.shape #data exploration


# In[4]:


df.dtypes#data exploration


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


df['full_text'] = df['text'] + df['flavor_text'] ##preparing predictor variable


# In[11]:


x = df['full_text'].fillna('') #handling nulls
x.isnull().any()


# In[12]:


y = df['color_identity'] ##defining y for multiclass 
y = [list(i)[0] if len(i) == 1 else -1 for i in y] ##-1 for more than one color


# In[16]:


# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder() 
y = label_encoder.fit_transform(y)


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x, y) ##splitting the data


# In[19]:


##model pipeline
text_clf = Pipeline([
('vect', CountVectorizer(input='content',ngram_range=(1,1),max_df=25, min_df=5)), ##text preprocessing
('tfidf', TfidfTransformer()), ##tfidf
('clf', LinearSVC())]) #model


# In[20]:


#fitting the data
text_clf.fit(x_train, y_train)


# In[21]:


pickle.dump(text_clf, open('multiclass.sav', 'wb')) #saving the model as object


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




