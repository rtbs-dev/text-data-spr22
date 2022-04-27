#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Dependencies
import pandas as pd 
from bertopic import BERTopic
import pickle


# In[2]:



#loading data
df = (pd.read_feather('C:/Users/VIOLIN/Desktop/text-data-spr22/data/mtg.feather')[['name','text', 'mana_cost', 'flavor_text','release_date', 'edhrec_rank']])


# In[3]:


df.shape #data exploration


# In[4]:


df.isnull().any() #data exploration


# In[5]:


df.isnull().sum()  #data exploration


# In[6]:


df = df.dropna() #data exploration


# In[7]:


df.isnull().any() #data exploration


# In[8]:


df.shape #data exploration


# In[9]:


topic_model = BERTopic() #model


# In[11]:


flavor_list = df["flavor_text"].to_list() #converting the text into list


# In[ ]:


topics, probs = topic_model.fit_transform(flavor_list)


# In[ ]:


topic_model.get_topic_info()


# In[ ]:


topic_model.visualize_topics() 


# In[ ]:


pickle.dump(text_clf, open('topic_model.sav', 'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




