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

def regression(seed=2022):
    """
    Prepares MTG data (X, y) and exports regression model using ElasticNetCV
    :param seed: Seed to guarantee consistent results
    """
    import pandas as pd

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split

    from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso

    
    import pickle
    
    df = pd.read_feather("../../../data/mtg.feather")
    
    # Remove observations without rank
    df = df[df['edhrec_rank'].notna()]
    
    df['text_flavor_text'] = df['text'] + df['flavor_text'].fillna('')

    tfidf = TfidfVectorizer(
        min_df=5, 
        stop_words='english')

    X_tfidf = tfidf.fit_transform(df['text_flavor_text'])

    ## Attempts to add non-text data to features; Failed to create CSR with added columns
    #from scipy.sparse import hstack
    #X_features = hstack((X_tfidf,np.array(df['converted_mana_cost'])[:,None])).toarray()
    
    y = df['edhrec_rank']
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.20, random_state=seed)
    
    elasticnet_model = ElasticNet(random_state=0)
    
    elasticnet_model.fit(X_train, y_train)
    
    pickle.dump(elasticnet_model, open('regression_elasticnet.sav', 'wb'))
    
    ## ElasticNetCV with 5 Cross Folds takes 18 minutes to run with identical results
    #regression_cv_model = ElasticNetCV(cv=5, random_state=0)
    #regression_cv_model.fit(X_train, y_train)
    #pickle.dump(regression_model, open('regression_elasticnet_cv.sav', 'wb'))
    
    lasso_model = Lasso(alpha=0.1)
    
    lasso_model.fit(X_train, y_train)
    
    pickle.dump(lasso_model, open('regression_lasso.sav', 'wb'))

if __name__ == "__main__":
    regression()


