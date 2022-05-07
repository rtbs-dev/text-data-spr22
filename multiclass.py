#!/usr/bin/env python
# coding: utf-8

# In[27]:


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
import sklearn.metrics as metrics
import yaml


# In[28]:


##Loading data
df = (pd.read_feather('C:/Users/VIOLIN/Desktop/text-data-spr22/data/mtg.feather')[['name','text', 'mana_cost', 'flavor_text','release_date', 'edhrec_rank','color_identity']])
df


# In[29]:


df.shape #data exploration


# In[30]:


df.dtypes #data exploration


# In[31]:


df.isnull().any() #data exploration


# In[32]:


df.isnull().sum()  #data exploration


# In[33]:


df = df.dropna() #data exploration


# In[34]:


df.isnull().any() #data exploration


# In[35]:


df.shape #data exploration


# In[36]:


df['full_text'] = df['text'] + df['flavor_text'] ##preparing predictor variable


# In[37]:


x = df['full_text'].fillna('') #handling nulls
x.isnull().any()


# In[38]:


y = df['color_identity'] ##defining y for multiclass 
y = [list(i)[0] if len(i) == 1 else -1 for i in y] ##-1 for more than one color


# In[39]:


# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder() 
y = label_encoder.fit_transform(y)


# In[40]:


x_train, x_test, y_train, y_test = train_test_split(x, y) ##splitting the data


# In[41]:


with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)


# In[42]:


ngrams = params["preprocessing"]["ngrams"]


# In[43]:


##model pipeline
text_clf = Pipeline([
('vect', CountVectorizer(input='content',ngram_range=(ngrams["smallest"], ngrams["smallest"]),max_df=25, min_df=5)), ##text preprocessing
('tfidf', TfidfTransformer()), ##tfidf
('clf', LinearSVC())]) #model


# In[44]:


#fitting the data
text_clf.fit(x_train, y_train)


# In[45]:


pickle.dump(text_clf, open('multiclass.sav','wb')) #saving the model as object


# #### Test data 

# In[46]:


y_pred = text_clf.predict(x_test)


# #### Mean Accuracy,Precision,Recall,F-Scores

# In[47]:


text_clf.score(x_test, y_test, sample_weight=None) ##mean accuracy


# In[48]:


metrics = pd.DataFrame(metrics.classification_report(y_test,y_pred,output_dict = True))
metrics


# In[49]:


metrics["weighted avg"].to_json("metrics1.json") ##printing both the avgs


# In[50]:


metrics["macro avg"].to_json("metrics2.json") #macro avg


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




