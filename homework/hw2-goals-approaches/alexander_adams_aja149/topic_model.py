#!/usr/bin/env python
# coding: utf-8
# %%

# %%


from bertopic import BERTopic
import pandas as pd

#Read in data, drop NAs, reset index
mtg = (pd.read_feather('../../../data/mtg.feather')).dropna(subset = ['flavor_text']).reset_index(drop=True)
#Instantiate BERT model
topic_model = BERTopic()
#Fit model to flavor text
topics, probs = topic_model.fit_transform(mtg['flavor_text'])
#Save model
topic_model.save('hw2_bert_model')

