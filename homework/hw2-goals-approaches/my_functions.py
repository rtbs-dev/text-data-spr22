from tqdm import tqdm
import re
import numpy as np

# Decontract text
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", str(phrase))
    phrase = re.sub(r"can\'t", "can not", str(phrase))
    # general
    phrase = re.sub(r"n\'t", " not", str(phrase))
    phrase = re.sub(r"\'re", " are", str(phrase))
    phrase = re.sub(r"\'s", " is", str(phrase))
    phrase = re.sub(r"\'d", " would", str(phrase))
    phrase = re.sub(r"\'ll", " will", str(phrase))
    phrase = re.sub(r"\'t", " not", str(phrase))
    phrase = re.sub(r"\'ve", " have", str(phrase))
    phrase = re.sub(r"\'m", " am", str(phrase))
    return phrase

# Preprocess text
def preprocess(text_column):
    my_list = []
    # tqdm is for printing the status bar
    for sentance in tqdm(text_column.values):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split())
        my_list.append(sent.strip())
    
    return my_list

# Multilabel Binarizer for columns
def multi(df, column_list):
    from sklearn.preprocessing import MultiLabelBinarizer
    import pandas as pd
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame()
    
    for i in column_list:
        multi_matrix = pd.DataFrame(mlb.fit_transform(df[i]),
                           columns=mlb.classes_,
                           index=df[i].index)
        res = pd.concat([res, multi_matrix], axis = 1, sort = False)
    return res     

