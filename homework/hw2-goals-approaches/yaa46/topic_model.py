# load packages
import pandas as pd
from bertopic import BERTopic

# load the data
mtg = list(pd.read_feather('../../../data/mtg.feather')["flavor_text"].values.dropna())

# accoridng to their githb you dont' really need to lowercase the text etc many of the regular preprocessing techniques
# Link: 

# setting the number of topic to auto automatically reduces topic using  hdbscan
# using 2 ngrams
# granted there are more names
# I alos tested using ngrams(1,2) but that came out terrible
# setting the number of topic to auto automatically reduces topic using  hdbscan
topic_model = BERTopic(nr_topics="auto", verbose=True, n_gram_range=(2,2))
topics, probs = topic_model.fit_transform(mtg)

# these is without the ngrams- as a noob this seems a little less inteerpretable
# topic_model = BERTopic(nr_topics="auto", verbose=True)
# topics, probs = topic_model.fit_transform(mtg)

# print the topics
topic_model.get_topic_info()
# topic -1 is outliers we can ignore that one

# reduce topics
topic_model.reduce_topics(mtg, topics, probs)

# get the new topics
topic_model.get_topic_info()

# find the topic
topic_model.find_topics("fight")
# this returns ___. We can use this to insepct our topics and sanity check the model

# look at the docs in the topic
topic_model.get_representative_docs(2)

# try the vectorizer model
# import count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

# when we vectorize it becomes even more interpretable
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
topic_model.update_topics(mtg, topics, vectorizer_model=vectorizer_model)

# get topics and probabilities
topics, probs = topic_model.fit_transform(mtg)

# save the model
topic_model.save("my_model")