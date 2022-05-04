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
import numpy as np
from bertopic import BERTopic
import nltk
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
```

```{code-cell} ipython3
df = (pd.read_feather('../../../data/mtg.feather')
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
# Import the topic model

topic_model = BERTopic.load('flavor_model')
```

```{code-cell} ipython3
# We need to recreate the probabilities and topics here, since they aren't saved with the model

probs = topic_model.hdbscan_model.probabilities_
topics = topic_model._map_predictions(topic_model.hdbscan_model.labels_)
```

```{code-cell} ipython3
# Visualize!

topic_model.visualize_topics()
```

```{code-cell} ipython3
# Let's reduce the number of topics to 7, since we see approximately 7 rough clusters.

flavor = df['flavor_text'].to_list()

new_topics, new_probs = topic_model.reduce_topics(flavor, topics, nr_topics=7)

topic_model.visualize_topics()
```

```{code-cell} ipython3
# Let's look at our new topics

topic_model.get_topic_info()
```

### Topics
1. Balduvia (https://mtg.fandom.com/wiki/Balduvia, https://mtg.fandom.com/wiki/LovisLiga_Coldeyes)
2. Maritime
3. Light/Dark
4. Mortality
5. Sword
6. Goblins
7. Predator/Prey

```{code-cell} ipython3
# Now we'll look at topics over time

timestamps = df.release_date.to_list()

topics_over_time = topic_model.topics_over_time(flavor, new_topics, timestamps)
topic_model.visualize_topics_over_time(topics_over_time, topics = [0, 1, 2, 3, 4, 5, 6])
```

If we look at topics over time it's a bit of a mess, though there are some jumps around 1995, and closer to 2020. There's a big jump around 1995 in the Balduvia topic, which corresponds with the release of the Ice Age set (https://mtg.fandom.com/wiki/Ice_Age). I assume that other similar spikes (though none happens over a broad a period of time) correspond with releases of themed sets. There also is a general rise in freuency of all themes after 2020. This may be related to an increase in sales and more cards being produced during the COVID-19 pandemic (https://wegotthiscovered.com/gaming/magic-gathering-revenues-significantly-2020-coronavirus-concerns/). This visualization looks at gross frequency though - what if we look at the relative importance of themes over time?

```{code-cell} ipython3
# Generate cards per release date

counts = pd.DataFrame(df.release_date.value_counts())
counts = counts.reset_index() 
counts.columns = ['Timestamp', 'Count']
```

```{code-cell} ipython3
# Merge with the topics_over_time dataframe and generate the frequency of a given topic relative to the number of cards released

topics_over_time = topics_over_time.merge(counts, on='Timestamp')
topics_over_time['Frequency'] = topics_over_time['Frequency'] / topics_over_time['Count']
```

```{code-cell} ipython3
# Visualize relative frequency of topics over time

topic_model.visualize_topics_over_time(topics_over_time, topics = [0, 1, 2, 3, 4, 5, 6])
```

Looking at relative importance of themes over time, we see some severe spikes (probably due to just a small number of cards released on a given day - it is unlikely that 100% of cards released in a set would be related to one topic. We still see the importance of Balduvia with the release of the Ice Age set in 1995, and a rise in the frequency of theme 5 (Goblins) in the past couple of years. In order to account for variance in how cards are released (with some dates only having one card releaesd and others having over 1,000), it may be useful to look at the importance of themes by month rather than by date of release. It's also interesting the topic 1 (Balduvia) seems to be the only one related to a specific fictional setting - I wonder if this is because BERTopic missed the others, other settings simply aren't as prominent, or if BERTopic combined them all within a fantasy setting topic (rather than a Balduvia topic). 

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
# Load up the two models

mcobject = open('multiclass_model.pkl', 'rb')
mcmodel = pickle.load(mcobject)

mlobject = open('multilabel_model.pkl', 'rb')
mlmodel = pickle.load(mlobject)
```

```{code-cell} ipython3
# Prep our data for predicted values

df1 = (pd.read_feather('../../../data/mtg.feather', 
                      columns = ['name','text', 'color_identity','flavor_text', 'release_date']
                     )
                     ).dropna(subset=['color_identity', 'text', 'flavor_text'])

df1['color_identity'] = df1.color_identity.where(df1.color_identity.str.len() == 1, np.nan)
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
