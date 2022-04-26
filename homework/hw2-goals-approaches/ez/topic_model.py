"""Codes for unsupervised exploration."""

# Packages
import pandas as pd
from bertopic import BERTopic
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the data
df = pd.read_feather("../../../data/mtg.feather")
docs = df["flavor_text"].dropna().to_list()

# Train a BERTopic model
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

topic_model.save("ft_model")
