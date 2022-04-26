"""Codes for multiclass model"""

# packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle


# Load the data
df = pd.read_feather("../../../data/mtg.feather")

# Remove NAs
df = df.dropna(subset = ["flavor_text", "text", "color_identity"]).reset_index(drop=True)

# Remove multicolored cards
df = df[df.color_identity.map(len) < 2]

# X
tfidf = TfidfVectorizer(
    min_df=5, # ignore rare words (appear in less than 5 documents)
    max_df=0.8, # ignore common words (appear in more than 80% of documents)
    stop_words="english"
)
text = df["text"] + df["flavor_text"].fillna('')
X = tfidf.fit_transform(text)

# y
ci = [list(i)[0] if len(i) == 1 else 0 for i in df.color_identity]
le = preprocessing.LabelEncoder()
y = le.fit_transform(ci)

# Instantiate LinearSVC
multiclass_model = LinearSVC()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=111)

# Fit LinearSVC
multiclass_model.fit(X_train, y_train)

# Save the model using pickle
pickle.dump(multiclass_model, open('multiclass_model.sav', 'wb'))
