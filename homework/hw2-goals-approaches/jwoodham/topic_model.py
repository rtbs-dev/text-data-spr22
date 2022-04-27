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
from bertopic import BERTopic
df = (pd.read_feather('C:/Users/JeffW/text-data-spr22/data/mtg.feather', 
                      columns = ['name','text', 'mana_cost', 'flavor_text','release_date', 'edhrec_rank']
                     )
                     ).dropna(subset=['flavor_text'])

# +
from bertopic import BERTopic

topic_model = BERTopic()

topics, probs = topic_model.fit_transform(df['flavor_text'].tolist())
# -

topic_model.get_topic_info()

topic_model.get_topic(0)

topic_model.save('flavor_model')
