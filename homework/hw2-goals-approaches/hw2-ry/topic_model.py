#packages
import pandas as pd

from tqdm.notebook import tqdm as notebook_tqdm
from bertopic import BERTopic
#data

df = (pd.read_feather('../../../data/mtg.feather')[['text',  'flavor_text','release_date']])
docs = df['flavor_text'].dropna().tolist()

#model training
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

#save model
topic_model.save("topic_model")
