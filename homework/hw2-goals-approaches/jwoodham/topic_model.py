# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python [conda env:text-data-class]
#     language: python
#     name: conda-env-text-data-class-py
# ---

import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import janitor as pj
from bertopic import BERTopic
df = (pd.read_feather('../../../data/mtg.feather', 
                      columns = ['name','text', 'colors', 'flavor_text','release_date', 'edhrec_rank']
                     )
                     ).dropna(subset=['flavor_text'])

# +
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN

vectorizer_model = CountVectorizer(ngram_range=(1,1), stop_words="english")

hdbscan_model = HDBSCAN(min_cluster_size=50)

topic_model = BERTopic(nr_topics = 'auto', vectorizer_model = vectorizer_model, hdbscan_model=hdbscan_model)

topics, probs = topic_model.fit_transform(df['flavor_text'].to_list())
# -

topic_model.get_topic_info()

topic_model.get_topic(0)

topic_model.save('flavor_model')
