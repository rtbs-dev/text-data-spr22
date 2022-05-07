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

# Homework 2/3

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
from bertopic import BERTopic
```

```{code-cell} ipython3
# Load the BERTopic model
ft_model = BERTopic.load("ft_model")
```

```{code-cell} ipython3
# Visualize topics
ft_model.visualize_topics(top_n_topics = 9)
```

![ft_model.png](attachment:ft_model.png)

```{code-cell} ipython3
# Get topic information
ft_model.get_topic_info()[2:10].set_index("Topic")
```

#### Renaming Topics
Topic 1: kami <br>
Kami is the Japanese word for "great spirits" and kamigawa means "river of the gods." This topic contains all things about gods and their world. <br>
<br>
Topic 2: goblin <br>
Goblin is a type of creature in mtg and Squee is a quite common name for goblin. This topic is about the goblin creature.<br>
<br>
Topic 3: Garruk Wildspeaker <br>
Garruk Wildspeaker is a planewalker. Since planewalkers are considered the most powerful beings in mtg, many flavor texts may refer to them.<br>
<br>
Topic 4: dragon <br>
Dragon is a type of creature in mtg.<br>
<br>
Topic 5: darkness <br>
Many of the mtg stories would mention dark magic of wizards, skeletons, etc. This topic is probably all about the dark side.<br>
<br>
Topic 6: gerrard <br>
Gerrard Capashen is the protagonist of the mtg storyline. The "legacy" word appears in this topic because Garrard is the heir to Urza, a famous planewalker. This topic is telling the story around Garrard.<br>
<br>
Topic 7: mage <br>
Since mtg is all about magic, it makes sense that mage would appear in various flavor texts.<br>
<br>
Topic 8: instant/socery <br>
After some research, I found that temper (Fiery Temper), geyser (Mana Geyser) and refract (refraction trap) are all among the instant or sorcery cards.<br>

```{code-cell} ipython3
# Prepare data
df_notna = df.dropna(subset=["flavor_text", "release_date"]).reset_index(drop=True)
docs = df_notna.flavor_text.to_list()
timestamps = df_notna.release_date.to_list()
```

```{code-cell} ipython3
topics, probs = ft_model.fit_transform(docs)
topics_over_time = ft_model.topics_over_time(docs, topics, timestamps)
```

```{code-cell} ipython3
# Visualize topics over time
ft_model.visualize_topics_over_time(topics_over_time, top_n_topics=9)
```

![topics_over_time.png](attachment:topics_over_time.png)

+++

From the figure, we can see that the kami topic peaks around 2004, suggesting that there was probably an expansion set about kami released in 2004. After some research, I found out that *Champions of Kamigawa* was released in October 2004 as the first set in the Kamigawa block and it introduced many rare creatures with kami-like magics. I actually started playing mtg recently, and bought the *Kamigawa: Neon Dynasty* expansion set (released in February 2022) over the weekend. According to the storyline, this set represents the current era on Kamigawa, which is more than 1200 years after conclusion of the original Kamigawa block. If we excluded the 2004 kami outlier, we would probably see another smaller spike in 2022.

```{code-cell} ipython3
# Visualize topics over time without the 'kami' outlier
ft_model.visualize_topics_over_time(topics_over_time, topics=[1,3,4,5,6,7])
```

![topics_over_time_no_kami.png](attachment:topics_over_time_no_kami.png)

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

+++

### Multiclass

```{code-cell} ipython3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
```

```{code-cell} ipython3
# Load the multiclass model
multiclass_model = pickle.load(open("multiclass.sav", 'rb'))
```

Preprocessing:
- Remove NAs in column _flavor_text_, _text_, and _color_identity_
- Remove multicolored _color_identity_
- min_df = 5, ignore rare words (appear in less than 5 documents)
- max_df = 0.8, ignore common words (appear in more than 80% of documents)

```{code-cell} ipython3
from multiclass import X, y
```

```{code-cell} ipython3
ConfusionMatrixDisplay.from_estimator(multiclass_model, X, y)
plt.show()
```

### Multilabel

```{code-cell} ipython3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import multilabel_confusion_matrix
```

```{code-cell} ipython3
# Load the multilabel model
multilabel_model = pickle.load(open("multilabel.sav", 'rb'))
```

Preprocessing:
- min_df = 5, ignore rare words (appear in less than 5 documents)
- max_df = 0.8, ignore common words (appear in more than 80% of documents)
- Transform _color_identity_ into lowecase

```{code-cell} ipython3
from multilabel import X_test, y_test
```

```{code-cell} ipython3
y_pred = multilabel_model.predict(X_test)
multilabel_confusion_matrix(y_test, y_pred)
```

## Part 3: Regression?

> Can we predict the EDHREC "rank" of the card using the data we have available? 

- Like above, add a script and dvc stage to create and train your model.
- In the notebook, aside from your descriptions, plot the `predicted` vs. `actual` rank, with a 45-deg line showing what "perfect prediction" should look like. 
- This is a freeform part, so think about the big picture and keep track of your decisions: 
    - what model did you choose? Why? 
    - What data did you use from the original dataset? How did you proprocess it? 
    - Can we see the importance of those features? e.g. logistic weights? 
    
How did you do? What would you like to try if you had more time?

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from plotnine import *
```

```{code-cell} ipython3
# Load the regression model
reg_model = pickle.load(open("regression.sav", 'rb'))
```

I choose to perform a regression model, and feed in the following as X:
- converted_mana_cost
- power
- toughness
- rarity (create a dummy variable for each rarity level)
- keywords (create a dummy variable of whether the card has any keywords)
- color_identity (create a dummy variable for each color)
- types (create a dummy variable for each type)

```{code-cell} ipython3
from regression import X, y, X_train, y_train, X_test, y_test
```

```{code-cell} ipython3
# Regression model score
reg_model.score(X_test, y_test)
```

```{code-cell} ipython3
plt.plot(y, y, color='k', ls='--')
plt.scatter(y_test, reg_model.predict(X_test), alpha=0.5, s=1)

plt.title('regression actual vs Pred.')
```

## Extra Credit (5 pts)

```{code-cell} ipython3
vi = permutation_importance(reg_model, X_train, y_train, n_repeats=5)
```

```{code-cell} ipython3
# Organize as a df
vi_df = pd.DataFrame(dict(variable=X_train.columns,
                          vi = vi['importances_mean'],
                          std = vi['importances_std']))

# Generate intervals
vi_df['low'] = vi_df['vi'] - 2*vi_df['std']
vi_df['high'] = vi_df['vi'] + 2*vi_df['std']

# Put in order from most to least important
vi_df = vi_df.sort_values(by="vi", ascending=False).reset_index(drop=True)
```

```{code-cell} ipython3
top10 = vi_df[0:10]

# Plot
(
    ggplot(top10,
           aes(x="variable",y="vi")) +
    geom_col(alpha=.5) +
    geom_point() +
    geom_errorbar(aes(ymin="low", ymax="high"), width=.2) +
    scale_x_discrete(limits=top10.variable.tolist()) +
    coord_flip() +
    labs(y="Reduction in AUC ROC",x="")
)
```

From the variable importance plot, we can see that `rarity` is the most important of all variables. It makes sense, because the rarer a card is, the higher it ranks. `converted_mana_cost` ranks the second, suggesting that the more mana a card needs, the stronger. `power` and `toughness` play a big part because they basically represent how strong a card attacks and blocks. The `keywords` variable here is a dummy on whether the card has keywords. In mtg, if a card has some keywords, it usually means that it has extra abilities, such as flying, haste, etc. It seems that the `color-identity` of a card doesn't matter that much, which is understandable, because there are strong cards in every color.

+++

## Part 4: Iteration, Measurement, & Validation

> No model is perfect, and experimentation is key. How can we more easily iterate and validate our model? 

- Pick **ONE** of your models above (regression, multilabel, or multiclass) that you want to improve or investigate, and calculate metrics of interest for them to go beyond our confusion matrix/predicted-actual plots:
    - for multiclass, report average and F1
    - for multilabel, report an [appropriate metric](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics) (e.g. `ranking_loss`)
    - for regression, report an [appropriate metric](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) (e.g. 'MAPE' or MSE), **OR** since these are ranks, the [pearson correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html) between predicted and actual may be more appropriate?
- in the corresponding `dvc.yaml` stage for your model-of-interest, add `params` and `metrics`
    - under `params`, pick a setting in your preprocessing (e.g. the `TfidfVecorizer`) that you want to change to imrpove your results. Set that param to your current setting, and have your model read from a `params.yaml` rather than directly writing it in your code.
    - under `metrics`, reference your `metrics.json` and have your code write the results as json to that file, rather than simply printing them or reporting them in the notebook.
- commit your changes to your branch, run `dvc repro dvc.yaml` for your file, then run a new experiment that changes that one parameter: e.g. `dvc exp run -S preprocessing.ngrams.largest=1` (see the `example/` folder for a complete working example).

Report the improvement/reduction in performance with the parameter change for your metric, whether by copy-pasting or using `!dvc exp diff` in the notebook, the results of `dvc exp diff`.

```{code-cell} ipython3
from multiclass_pipe import X_train, X_test, y_train, y_test
from sklearn.metrics import classification_report
```

```{code-cell} ipython3
# Load the multiclass_pipe model
multiclass_pipe = pickle.load(open("multiclass_pipe.sav", 'rb'))
```

```{code-cell} ipython3
y_pred = multiclass_pipe.predict(X_test)
print(classification_report(y_test, y_pred))
```

```{code-cell} ipython3
# dvc exp run -S preprocessing.ngrams.largest=3
!dvc exp diff
```
