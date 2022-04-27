#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Dependencies
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
### model

from sklearn.model_selection import train_test_split
from sklearn import linear_model


# In[2]:


df = (pd.read_feather('C:/Users/VIOLIN/Desktop/text-data-spr22/data/mtg.feather')[['name','text', 'mana_cost', 'flavor_text','release_date', 'edhrec_rank','color_identity']])
df #load data


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


df['full_text'] = df['text'] + df['flavor_text'] #predictor


# In[10]:


x = df['full_text'].fillna('') #handling nulls
x.isnull().any()


# In[11]:


y = df["edhrec_rank"] #y 


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(x, y) #splitting the data


# In[13]:


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train)
X_train_counts.shape


# In[14]:


tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts) ##tfidf transformation
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape


# In[15]:


## Filtering thr first 30 columns 
X_train_counts = X_train_counts[ :,:30]
X_train_counts.shape


# In[16]:



clf = linear_model.Lasso(alpha=0.1) ##applying lasso regression


# In[ ]:


clf.fit(X_train_tf,y_train) #fitting data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




