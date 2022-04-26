---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python [conda env:text-data-class]
  language: python
  name: conda-env-text-data-class-py
---

# Homework 2

> The body grows stronger under stress. The mind does not.
> 
>  -- Magic the Gathering, _Fractured Sanity_

+++

## Setup
### Packages

```{code-cell} ipython3
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
```

```{code-cell} ipython3
from bertopic import BERTopic
```

```{code-cell} ipython3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
```

### Data
Run `dvc pull` to ensure my local copy of the repo has the actual data

```{code-cell} ipython3
!dvc pull
```

Load the data using pandas+pyarrow:

```{code-cell} ipython3
df = pd.read_feather('../../../data/mtg.feather')
df.head()[['name','text', 'mana_cost', 'flavor_text','release_date', 'edhrec_rank']]
```

### Submission Structure
You will need to submit a pull-request on DagsHub with the following additions: 

- your subfolder, e.g. named with your user id, inside the `homework/hw2-goals-approaches/` folder
    - your "lab notebook", as an **`.ipynb` or `.md`** (e.g. jupytext), that will be exported to PDF for Canvas submission. **This communicates your _goals_**, along with the results that will be compared to them. 
    - your **`dvc.yaml`** file that will define  the inputs and outputs of your _approaches_. See [the DVC documentation](https://dvc.org/doc/user-guide/project-structure/pipelines-files) for information!
    - **source code** and **scripts** that define the preprocessing and prediction `Pipeline`'s you wish to create. You may then _print_ the content of those scripts at the end of your notebook e.g. as appendices using 
- any updates to `environment.yml` to add the dependencies you want to use for this homework

+++

## Part 1: Unsupervised Exploration

Investigate the [BERTopic](https://maartengr.github.io/BERTopic/index.html) documentation (linked), and train a model using their library to create a topic model of the `flavor_text` data in the dataset above. 

- In a `topic_model.py`, load the data and train a bertopic model. You will `save` the model in that script as a new trained model object
- add a "topic-model" stage to your `dvc.yaml` that has `mtg.feather` and `topic_model.py` as dependencies, and your trained model as an output
- load the trained bertopic model into your notebook and display
    1. the `topic_visualization` interactive plot [see docs](https://maartengr.github.io/BERTopic/api/plotting/topics.html)
    2. Use the plot to come up with working "names" for each major topic, adjusting the _number_ of topics as necessary to make things more useful. 
    3. Once you have names, create a _Dynamic Topic Model_ by following [their documentation](https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html). Use the `release_date` column as timestamps. 
    4. Describe what you see, and any possible issues with the topic models BERTopic has created. **This is the hardest part... interpreting!**

```{code-cell} ipython3
# Load the BERTopic model
ft_model = BERTopic.load("ft_model")
```

```{code-cell} ipython3
# Visualize topics
ft_model.visualize_topics(top_n_topics = 9)
```

![dt_model.png](attachment:dt_model.png)

```{code-cell} ipython3
# Get topic information
ft_model.get_topic_info()[2:10].set_index("Topic")
```

#### Renaming Topics
Topic 1: kami <br>
Topic 2: goblin <br>
Topic 3: wildspeaker <br>
Topic 4: dragon <br>
Topic 5: darkness <br>
Topic 6: gerrard <br>
Topic 7: mage <br>
Topic 8: temper <br>

```{code-cell} ipython3
# Prepare data
df_notna = df.dropna(subset=["flavor_text", "release_date"])
docs = df_notna.flavor_text.to_list()
timestamps = df_notna.release_date.to_list()
```

```{code-cell} ipython3
# Train a BERTopic model
topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(docs)

topics_over_time = topic_model.topics_over_time(docs, topics, timestamps)
```

```{code-cell} ipython3
# Visualize topics over time
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=9)
```

![topics_over_time.png](attachment:topics_over_time.png)

+++

Kami is the Japanese word for "great spirits." From the figure, we can see that the kami topic peaks around 2004, suggesting that there was probably an expansion set about kami released in 2004. After some research, I found out that *Champions of Kamigawa* was released in October 2004 as the first set in the Kamigawa block and it introduced many rare creatures with kami-like magics. I actually started playing mtg recently, and bought the *Kamigawa: Neon Dynasty* expansion set (released in February 2022) over the weekend. According to the storyline, this set represents the current era on Kamigawa, which is more than 1200 years after conclusion of the original Kamigawa block. If we excluded the 2004 kami outlier, we would probably see another smaller spike in 2022.

```{code-cell} ipython3
# Visualize topics over time without the 'kami' outlier
topic_model.visualize_topics_over_time(topics_over_time, topics=[0,1,3,4,5,6,7])
```

![topics_over_time_no_kami-2.png](attachment:topics_over_time_no_kami-2.png)

+++

After removing the kami outlier from the figure, we can see that the goblin topic peaks arouund 2012, 2016 and 2022, suggesting that goblin cards probably have gained popularity, resulting in multiple goblin-related expansion sets released over the years. This makes great sense to me, because I think that goblin cards are pretty strong and personally love using them.

+++

## Part 2 Supervised Classification

Using only the `text` and `flavor_text` data, predict the color identity of cards: 

Follow the sklearn documentation covered in class on text data and Pipelines to create a classifier that predicts which of the colors a card is identified as. 
You will need to preprocess the target _`color_identity`_ labels depending on the task: 

- Source code for pipelines
    - in `multiclass.py`, again load data and train a Pipeline that preprocesses the data and trains a multiclass classifier (`LinearSVC`), and saves the model pickel output once trained. target labels with more than one color should be _unlabeled_! 
    - in `multilabel.py`, do the same, but with a multilabel model (e.g. [here](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_multilabel.html#sphx-glr-auto-examples-miscellaneous-plot-multilabel-py)). You should now use the original `color_identity` data as-is, with special attention to the multi-color cards. 
- in `dvc.yaml`, add these as stages to take the data and scripts as input, with the trained/saved models as output. 

- in your notebook: 
    - Describe:  preprocessing steps (the tokenization done, the ngram_range, etc.), and why. 
    - load both models and plot the _confusion matrix_ for each model ([see here for the multilabel-specific version](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html))
    - Describe: what are the models succeeding at? Where are they struggling? How do you propose addressing these weaknesses next time?

```{code-cell} ipython3
# Load the multiclass model
multiclass_model = pickle.load(open("multiclass.sav", 'rb'))
```

```{code-cell} ipython3
# Load the data
df = pd.read_feather("../../../data/mtg.feather")

# Remove NAs
df = df.dropna(subset = ["flavor_text", "text", "color_identity"]).reset_index(drop=True)

# Remove multicolored cards
df = df[df.color_identity.map(len) < 2]
```

```{code-cell} ipython3
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=111)
```

```{code-cell} ipython3
plot_confusion_matrix(multiclass_model, X_test, y_test)  
plt.show()
```

## Part 3: Regression?

> Can we predict the EDHREC "rank" of the card using the data we have available? 

- Like above, add a script and dvc stage to create and train your model
- in the notebook, aside from your descriptions, plot the `predicted` vs. `actual` rank, with a 45-deg line showing what "perfect prediction" should look like. 
- This is a freeform part, so think about the big picture and keep track of your decisions: 
    - what model did you choose? Why? 
    - What data did you use from the original dataset? How did you proprocess it? 
    - Can we see the importance of those features? e.g. logistic weights? 
    
How did you do? What would you like to try if you had more time?