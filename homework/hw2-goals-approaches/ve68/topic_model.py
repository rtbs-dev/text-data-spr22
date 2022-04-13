if __name__ == "__main__":
    import pandas as pd
    from bertopic import BERTopic

    df = pd.read_feather("../../../data/mtg.feather")

    flavor_text = df['flavor_text'].dropna().to_list()

    topic_model = BERTopic()

    topics, probs = topic_model.fit_transform(flavor_text)

    topic_model.save("flavor_text_topics")
else:
    print("Error")