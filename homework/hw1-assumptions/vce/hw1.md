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

**2022/02/17**


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

# regexr test
# "(^[A-Z]{1}[\w ]*):\\n([\w .,:;?!'\"]*)[\\n]+(?=[A-Z])"

patt = re.compile(
    "(^[A-Z]{1}[\w]*[ A-Z]?[\w]*):" # If there is a second word, make sure it starts with a capital letter
    "\\n(.*?)"
    "(?=\\n\\n|\Z)",
    flags=re.S | re.M
)
```

```{code-cell} ipython3
# FIND `patt` SUCH THAT:  
matches = patt.findall(content)
```

```{code-cell} ipython3
import pandas as pd

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
import requests

url_A_K = "https://en.wikipedia.org/wiki/List_of_Shakespearean_characters_(A%E2%80%93K)"
page_A_K = requests.get(url_A_K)

url_L_Z = "https://en.wikipedia.org/wiki/List_of_Shakespearean_characters_(L%E2%80%93Z)"
page_L_Z = requests.get(url_L_Z)
```

```{code-cell} ipython3
patt2 = re.compile(
    #"<ul><li><b>([\w ]*)</b>[\w\W]*?<i>([\w ]*)[\w\W]*?(?=\\n)",
    "<ul><li><b>([\w ]*)</b>[\w\W]*?>([\w ]*)</a>[\w\W]*?(?=\\n)",
    flags=re.S | re.M
)
```

```{code-cell} ipython3
matches_A_K = patt2.findall(page_A_K.text)

table_A_K = pd.DataFrame.from_records(matches_A_K, columns=['character','play'])
```

```{code-cell} ipython3
matches_L_Z = patt2.findall(page_L_Z.text)

table_L_Z = pd.DataFrame.from_records(matches_L_Z, columns=['character','play'])
```

```{code-cell} ipython3
table_A_Z = pd.concat([table_A_K,table_L_Z],ignore_index=True)
```

```{code-cell} ipython3
table_ts['speaker_lower'] = table_ts['speaker'].str.lower()
table_A_Z['character_lower'] = table_A_Z['character'].str.lower()
```

```{code-cell} ipython3
table = table_ts\
    .merge(table_A_Z, how='left', left_on='speaker_lower', right_on='character_lower')\
    .drop(columns = ['speaker_lower','character','character_lower'])
```

```{code-cell} ipython3
table['play'].value_counts()
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

```
