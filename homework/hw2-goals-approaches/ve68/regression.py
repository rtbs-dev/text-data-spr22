# ## Part 3: Regression
#
# > Can we predict the EDHREC "rank" of the card using the data we have available? 
#
# - Like above, add a script and dvc stage to create and train your model
# - in the notebook, aside from your descriptions, plot the `predicted` vs. `actual` rank, with a 45-deg line showing what "perfect prediction" should look like. 
# - This is a freeform part, so think about the big picture and keep track of your decisions: 
#     - What model did you choose? Why? 
#     - What data did you use from the original dataset? How did you proprocess it? 
#     - Can we see the importance of those features? e.g. logistic weights? 
#     
# How did you do? What would you like to try if you had more time?

def regression():
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    
    from sklearn.linear_model import ElasticNet
    
    from joblib import dump

    df = pd.read_feather("../../../data/mtg.feather")

    df_er_ft_rd = df[['edhrec_rank','flavor_text', 'text']].dropna().reset_index(drop=True)

    text = df_er_ft_rd['text'] + df_er_ft_rd ['flavor_text'].fillna('')

    tfidf = TfidfVectorizer(
        min_df=5, 
        stop_words='english')

    X_tfidf = tfidf.fit_transform(text)

    y = df_er_ft_rd ['edhrec_rank']
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, random_state = 20220420)
    
    regression_model = ElasticNet(random_state=0)

    regression_model.fit(X_train, y_train)
    
    dump(regression_model, 'regrsesion.joblib') 

if __name__ == "__main__":
    regression()
