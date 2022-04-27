#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import modules
import pandas as pd
import numpy as np
from bertopic import BERTopic


# In[2]:


# load in data
(pd.read_feather('C:/Georgetown University/Courses/Spring Semester 2022/Text As Data/text-data-spr22/data/mtg.feather')# <-- will need to change for your notebook location
 .head()  
)


# In[4]:


# store full data
df = (pd.read_feather('C:/Georgetown University/Courses/Spring Semester 2022/Text As Data/text-data-spr22/data/mtg.feather')  
)

# check shape
df.shape


# In[5]:


# remove rows where flavor-text col has missing values
df2 = df.dropna(how='any', subset= ['flavor_text'])

# check shape
df2.shape


# In[6]:


# store flavor_text data as list
flavor_text_list = df2.flavor_text.tolist()

# check length
len(flavor_text_list)


# In[7]:


# store topic model
topic_model = BERTopic(low_memory = True)

# fit topic model
topics, probs = topic_model.fit_transform(flavor_text_list)

# save topic model
topic_model.save("flav_text_model")

