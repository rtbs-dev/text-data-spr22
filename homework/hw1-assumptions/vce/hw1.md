---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.6
kernelspec:
  display_name: Python [conda env:root] *
  language: python
  name: conda-root-py
---

# Homework: _State Your Assumptions_ 
**PPOL 628**

**Vince Egalla (ve68)**

**2022/03/01**


> POLONIUS\
> _What do you read, my lord?_
> 
> HAMLET\
> _Words, words, words_
> 
>  -- _Hamlet_, Act 2, Scene 2

This homework deals with the assumptions made when taking text from its original "raw" form into something more _computable_.

- Assumptions about the _shape_ of text (e.g. how to break a corpus into documents)
- Assumptions about what makes a _token_, an _entity_, etc. 
- Assumptions about what interesting or important _content_ looks like, and how that informs our analyses.


There are three parts: 
1. Splitting Lines from Shakespeare
2. Tokenizing and Aligning lines into plays
3. Assessing and comparing characters from within each play

**NB**\
This file is merely a _template_, with instructions; do not feel constrained to using it directly if you do not wish to.

+++

## Get the Data

Since the class uses `dvc`, it is possible to get this dataset either using the command line (e.g. `dvc import https://github.com/TLP-COI/text-data-course resources/data/shakespeare/shakespeare.txt`), or using the python api (if you wish to use python)

+++

from dvc.api import read,get_url
import pandas as pd

txt = read('resources/data/shakespeare/shakespeare.txt', 
           repo='https://github.com/TLP-COI/text-data-course')

print(txt[:250])

```{code-cell} ipython3
# Open local version of shakespeare.txt
with open('shakespeare.txt', 'r') as f:
    content = f.read()
```

Make sure this works before you continue! 
Either way, it would likely be beneficial to have the data downloaded locally to keep from needing to re-dowload it every time.

+++

## Part 1

Split the text file into a _table_, such that 
- each row is a single _line_ of dialogue
- there are columns for
  1. the speaker
  1. the line number
  1. the line dialogue (the text)

_Hint_: you will need to use RegEx to do this rapidly. See the in-class "markdown" example!

Question(s): 
- What assumptions have you made about the text that allowed you to do this?

```{code-cell} ipython3
import re

patt = re.compile(
    
    # Captures speaker at start of line
        # Checks for capital letter
        # Checks for optional second capital letter for a second word
        # Will fail to capture speaker with three words (e.g. Second Class Citizen)
    
    "(^[A-Z]{1}[\w]*[ A-Z]?[\w]*):" 
    
    # Captures dialogue between speaker and end of line
    
    "\\n(.*?)"
    
    # Positive look ahead for end of line and end of text
    
    "(?=\\n\\n|\Z)",
    flags=re.S | re.M
)
```

```{code-cell} ipython3
# Find `patt` such that:  
matches = patt.findall(content)
```

```{code-cell} ipython3
import pandas as pd

# Convert matches to table with line number
table_ts = pd.DataFrame.from_records(matches, columns=['speaker','dialogue'])

table_ts['line_number'] = range(1, len(matches) + 1)
```

## Part 2

You have likely noticed that the lines are not all from the same play!
Now, we will add some useful metadata to our table: 

- Determine a likely source title for each line
- add the title as a 'play' column in the data table. 
- make sure to document your decisions, assumptions, external data sources, etc. 

This is fairly open-ended, and you are not being judged completely on _accuracy_. 
Instead, think outside the box a bit as to how you might accomplish this, and attempt to justify whatever approximations or assumptions you felt were appropriate.

```{code-cell} ipython3
# Check if wordnet is a viable method for determining metadata

import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.find('corpora/wordnet')
    nltk.download('omw-1.4')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet'); 
    

from nltk.corpus import wordnet as wn

wn.synsets('Shakespeare')

print(wn.synset('shakespeare.n.01').part_meronyms())
```

```{code-cell} ipython3
# Metadata capture method
    # Scrape Wikipedia for characters and play they appear in using regex
    # Match character to speaker to merge play

import requests

# Get text content of both Wikipedia pages
url_A_K = "https://en.wikipedia.org/wiki/List_of_Shakespearean_characters_(A%E2%80%93K)"
page_A_K = requests.get(url_A_K)

url_L_Z = "https://en.wikipedia.org/wiki/List_of_Shakespearean_characters_(L%E2%80%93Z)"
page_L_Z = requests.get(url_L_Z)
```

```{code-cell} ipython3
# Regex pattern for scraping character and play
patt2 = re.compile(
    #"<ul><li><b>([\w ]*)</b>[\w\W]*?<i>([\w ]*)[\w\W]*?(?=\\n)",
    "<ul><li><b>([\w ]*)</b>[\w\W]*?>([\w ]*)</a>[\w\W]*?(?=\\n)",
    flags=re.S | re.M
)

# Regex doesn't perfectly capture tags from HTML text due to differing formats
# Intended to match list <ul> to end of line \n
    # Capture character name in bolded letters <b> ... </b> AND
    # play name through HTML tags with <i> ... </i> or <a> ... </a>
    # However, play name is less standardized than character name
    # Patt2 yielded more results than other patterns
```

```{code-cell} ipython3
# Capture characters and play names in page for A-K and convert to table
matches_A_K = patt2.findall(page_A_K.text)

table_A_K = pd.DataFrame.from_records(matches_A_K, columns=['character','play'])
```

```{code-cell} ipython3
# Capture characters and play names in page for L-Z and convert to table
matches_L_Z = patt2.findall(page_L_Z.text)

table_L_Z = pd.DataFrame.from_records(matches_L_Z, columns=['character','play'])
```

```{code-cell} ipython3
# Stack tables into one
table_A_Z = pd.concat([table_A_K,table_L_Z],ignore_index=True)
```

```{code-cell} ipython3
# Create temporary lower case versions for better matching
table_ts['speaker_lower'] = table_ts['speaker'].str.lower()
table_A_Z['character_lower'] = table_A_Z['character'].str.lower()

# One alternative here is to use fuzzy matching for better match rates
```

```{code-cell} ipython3
# Merge left to add meta data
# Drop unnecessary columns
table = table_ts\
    .merge(table_A_Z, how='left', left_on='speaker_lower', right_on='character_lower')\
    .drop(columns = ['speaker_lower','character','character_lower'])
```

```{code-cell} ipython3
# Check values of play metadata
table['speaker'].value_counts()
```

Out of 6460 observations of dialogue, only a fraction have metadata of a play. Method and regex are flawed in multiple ways.

First, non-plays are captured (e.g. 'hist', 'bed trick', 'Ghost character'), identified by non-proper case phrases.

Second, merge rate is fairly low. This may have multiple reasons as I am using an exact merge. Names must be spelled identically, beyond being lower case. White space on either side may be an issue here.

Third, names will only be merged to one of possible names. Without knowledge of Shakespeare, I am unsure of if characters could share a name between plays, as unique individuals. This method assumes that all characters are uniquely identifiable by name.

Fourth, non-named characters with dialgoue will not have a merge as they will not appear in Shakespeare characters Wikipedia page (e.g. 'First Citizen', 'All', 'Lords'). Groups of characters will be treated similarly.

```{code-cell} ipython3
# table_ts['speaker'].unique()
```

## Part 3

Pick one or more of the techniques described in this chapter: 

- keyword frequency
- entity relationships
- markov language model
- bag-of-words, TF-IDF
- semantic embedding

make a case for a technique to measure how _important_ or _interesting_ a speaker is. 
The measure does not have to be both important _and_ interesting, and you are welcome to come up with another term that represents "useful content", or tells a story (happiest speaker, worst speaker, etc.)

Whatever you choose, you must
1. document how your technique was applied
2. describe why you believe the technique is a valid approximation or exploration of how important, interesting, etc., a speaker is. 
3. list some possible weaknesses of your method, or ways you expect your assumptions could be violated within the text. 

This is mostly about learning to transparently document your decisions, and iterate on a method for operationalizing useful analyses on text. 
Your explanations should be understandable; homeworks will be peer-reviewed by your fellow students.

```{code-cell} ipython3
speakers = table['speaker'].str.lower().unique()
```

```{code-cell} ipython3
entity_list = []

for i in range(0, len(table)):
    if words_re.search(table.iloc[i]['dialogue'].lower()):
        m = words_re.search(table.iloc[i]['dialogue'].lower())
        entity_list.append([i, table.iloc[i]['speaker'].lower(), m.group(0)])
```

```{code-cell} ipython3
entity_df = pd.DataFrame(entity_list, columns=['line_number', 'speaker', 'addressed'])
```

```{code-cell} ipython3
entity_df_unique = entity_df\
    .drop(columns=['line_number'])\
    .drop_duplicates()\
    .reset_index(drop=True)
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import networkx as nx
```

```{code-cell} ipython3
G =  nx.from_pandas_edgelist(
    entity_df_unique,
    source = 'speaker',
    target = 'addressed')
```

```{code-cell} ipython3
# Default layout
nx.draw_networkx(G)
```

```{code-cell} ipython3
# Custom layout to increase distance between nodes and figure size
plt.figure(figsize=(16,16)) 
pos = nx.spring_layout(G, k=0.75, iterations=50)
nx.draw_networkx(G, pos=pos, alpha=0.8)
plt.show()
```

```{code-cell} ipython3
speaker_count = pd.DataFrame(entity_df_unique['speaker'].value_counts()\
    .rename_axis('speaker').reset_index(name='count'))
```

```{code-cell} ipython3
speaker_count
```

```{code-cell} ipython3
from plotnine import ggplot, aes, geom_point, theme, element_text, labs
```

```{code-cell} ipython3
(
    ggplot(speaker_count) +
        geom_point(
            aes(
                x='speaker',
                y='count')) +
        labs(
            x = 'Speaker',
            y = 'Count') +
        theme(
             axis_text_x=element_text(
                 rotation=90, 
                 hjust=1),
             figure_size=(12, 6))
)
```

```{code-cell} ipython3
(
    ggplot(speaker_count[speaker_count['count'] > 15]) +
        geom_point(
            aes(
                x='speaker',
                y='count')) +
        labs(
            x = 'Speaker',
            y = 'Count') +
        theme(
             axis_text_x=element_text(
                 rotation=90, 
                 hjust=1),
             figure_size=(12, 6))
)
```

```{code-cell} ipython3

```
