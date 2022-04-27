import pandas as pd

from hdbscan import HDBSCAN
from bertopic import BERTopic

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from my_functions import preprocess

df = (
    pd.read_feather('../../data/mtg.feather')
    .dropna(subset=['flavor_text', 'text'])
)

# Preprocess flavor_text
docs = preprocess(df.flavor_text)

# Custom embedding on cleaned flavor_text
sentence_model = SentenceTransformer("all-mpnet-base-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=True)

# HDBSCAN model
hdbscan_model = HDBSCAN(min_cluster_size=100, metric='euclidean', 
                        cluster_selection_method='eom',
                        prediction_data=True, min_samples=10)

# Vectorizer model
vectorizer_model = CountVectorizer(ngram_range=(1,1), stop_words="english")

# Train BERTopic
topic_model = BERTopic(
    verbose = True, 
    vectorizer_model = vectorizer_model,
    nr_topics = 'auto',
    calculate_probabilities=True,
    hdbscan_model = hdbscan_model
    )

# Fit
topic_model.fit(docs, embeddings)

# Save
topic_model.save("my_model_embeddings")

exit()
