import pandas as pd
from bertopic import BERTopic
#!dvc pull
#import os
#!dir

#os.getcwd()
#os.chdir('text-data-spr22/homework/hw2-goals-approaches/hw-mk')
#os.chdir('Desktop/projects/text-data-spr22/homework/hw2-goals-approaches/hw-mk')
#os.chdir('../')

def flavor_text_topic():
'''
Bert topic model on magic the gathering flavor text.
'''
    df = pd.read_feather("mtg.feather")
    flavor_text = df['flavor_text'].dropna().to_list()
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(flavor_text)
    topic_model.save("flavor_text_topics")


if __name__ == "__main__":
flavor_text_topic()
