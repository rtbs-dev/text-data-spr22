---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.6
kernelspec:
  display_name: Python [conda env:text-data-class]
  language: python
  name: conda-env-text-data-class-py
---

# Homework: Goals & Approaches

> The body grows stronger under stress. The mind does not.
> 
>  -- Magic the Gathering, _Fractured Sanity_

This homework deals with the goals you must define, along with the approaches you deem necessary to achieve those goals. 
Key to this will be a focus on your _workflows_: 

- are they reproducible? 
- are they maintainable? 
- are they well-justified and communicated? 

This is not a "machine-learning" course, but machine learning plays a large part in modern text analysis and NLP. 
Machine learning, in-turn, has a number of issues tracking and solving issues in a collaborative, asynchronous, distributed manner. 

It's not inherently _wrong_ to use pre-configured models and libraries! 
In fact, you will likely be unable to create a set of ML algorithms that "beat" something others have spent 100's of hours creating, optimizing, and validating. 
However, to answer the three questions above, we need a way to explicitly track our decisions to use others' work, and efficiently _swap out_ that work for new ideas and directions as the need arises. 

This homework is a "part 1" of sorts, where you will construct several inter-related pipelines in a way that will allow _much easier_ adjustment, experimentation, and measurement in "part 2"

+++

## Setup

### Dependencies 
As before, ensure you have an up-to-date environment to isolate your work. 
Use the `environment.yml` file in the project root to create/update the `text-data-class` environment. 
> I expect any additional dependencies to be added here, which will show up on your pull-request. 

### Data
Once again, we have set things up to use DVC to import our data. 
If the data changes, things will automatically update! 
The data for this homework has been imported as `mtg.feather` under the `data/` directory at the top-level of this repository. 
In order to ensure your local copy of the repo has the actual data (instead of just the `mtg.feather.dvc` stub-file), you need to run `dvc pull`

```{code-cell} ipython3
!dvc pull
```

Then you may load the data into your notebooks and scripts e.g. using pandas+pyarrow:

```{code-cell} ipython3
import pandas as pd
df = (pd.read_feather('C:/Users/JeffW/text-data-spr22/data/mtg.feather', 
                      columns = ['name','text', 'mana_cost', 'flavor_text','release_date', 'edhrec_rank']
                     )
                     ).dropna(subset=['flavor_text'])
```

But that's not all --- at the end of this homework, we will be able to run a `dvc repro` command and all of our main models and results will be made available for your _notebook_ to open and display.

+++

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
timestamps = df.release_date.to_list()
flavor_text = df.flavor_text.to_list()
topics = pd.read_csv('topics.csv')
topics.columns = ['topics']
topics = topics.topics.to_list()
```

```{code-cell} ipython3
from bertopic import BERTopic

topic_model = BERTopic.load('flavor_model')
topic_model.visualize_topics(top_n_topics=10)
```

```{code-cell} ipython3
topics_over_time = topic_model.topics_over_time(flavor_text, topics, timestamps)
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
```

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
import pickle

mcobject = open('multiclass_model.pkl', 'rb')
mcmodel = pickle.load(mcobject)

mlobject = open('multilabel_model.pkl', 'rb')
mlmodel = pickle.load(mlobject)
```

```{code-cell} ipython3
df1 = (pd.read_feather('C:/Users/JeffW/text-data-spr22/data/mtg.feather', 
                      columns = ['name','text', 'mana_cost','flavor_text', 'release_date']
                     )
                     ).dropna(subset=['mana_cost', 'text', 'flavor_text'])

df1['card_num'] = df1.reset_index().index

df1 = df1.explode('mana_cost')

for i in range(10):
    df1 = df1.loc[df1["mana_cost"] != str(i)]

df1.mana_cost.value_counts()

count = df1.groupby(['card_num'])['mana_cost'].count()
count = pd.DataFrame(count)

df1 = pd.merge(df1, count, how='left', on=['card_num'], validate = 'm:1')

df1 = df1[(df1['mana_cost_y'] == 1)]

df1 = df1[['name', 'text', 'mana_cost_x', 'flavor_text', 'release_date']]

y_true = df1['mana_cost_x']
```

```{code-cell} ipython3
y_pred = mcmodel.predict(df1['flavor_text'])
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.show()
```

```{code-cell} ipython3

```

```{code-cell} ipython3
from sklearn.preprocessing import MultiLabelBinarizer

df1 = (pd.read_feather('C:/Users/JeffW/text-data-spr22/data/mtg.feather', 
                      columns = ['name','text', 'mana_cost','flavor_text', 'release_date']
                     )
                     ).dropna(subset=['mana_cost', 'text', 'flavor_text'])

y_pred = mlmodel.predict(df1['flavor_text'])

mlb = MultiLabelBinarizer(classes = ['B','C','G','R','S','U','W','X'])
y_true = mlb.fit_transform(df1['mana_cost'])
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay

multilabel_confusion_matrix(y_true, y_pred)

f, axes = plt.subplots(2, 4, figsize=(25, 15))
axes = axes.ravel()
for i in range(8):
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true[:, i],
                                                   y_pred[:, i]),
                                  display_labels=[0, i])
    disp.plot(ax=axes[i], values_format='.4g')
    disp.ax_.set_title(f'class {i}')
    if i<10:
        disp.ax_.set_xlabel('')
    if i%5!=0:
        disp.ax_.set_ylabel('')
    disp.im_.colorbar.remove()

plt.subplots_adjust(wspace=0.10, hspace=0.1)
f.colorbar(disp.im_, ax=axes)
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
