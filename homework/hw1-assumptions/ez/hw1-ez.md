---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python [conda env:root]
  language: python
  name: conda-root-py
---

# Homework: _State Your Assumptions_ 

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

```{code-cell} ipython3
from dvc.api import read,get_url
import pandas as pd

txt = read('resources/data/shakespeare/shakespeare.txt', 
           repo='https://github.com/TLP-COI/text-data-course')

print(txt[:250])
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
import pandas as pd
import numpy as np
import re
```

```{code-cell} ipython3
# Use the first 250 characters as a sample
sample = txt[:250]
sample
```

**Assumptions:**
- the speaker begins with a capitalized letter and ends with a colon
- the dialogue begins with \n (new line) and ends with \n\n

```{code-cell} ipython3
patt_try = re.compile(
    "(^[A-Z].+?):$"  # the speaker
    "\n{1}(.*?)\n{2}",  # the dialogue
    flags = re.S | re.M
)
```

```{code-cell} ipython3
# Find `patt_try` such that:  
matches_try = patt_try.findall(sample)
pd.DataFrame.from_records(matches_try, columns=['speaker', 'dialogue'])
```

```{code-cell} ipython3
# Check the last several lines to see whether they meet the assumptions
print(txt[-330:])
```

```{code-cell} ipython3
txt[-330:]
```

Here, the last line will be omitted according to the current assumptions because it ends with \n instead of \n\n.

```{code-cell} ipython3
# Fix the last line by adding an extra \n
new_line = '\n'
txt = txt + new_line
```

Another particular case that doesn't meet the assumptions is empty dialogue.

```{code-cell} ipython3
# Example of empty dialogue
print(txt[11225:11536])
```

My assumption is that the dialogue begins with \n and ends with \n\n (a total of three \n's). However, an empty dialoge comes with only two \n's.

```{code-cell} ipython3
# Fix empty dialogue by adding an extra \n
txt = txt.replace(':\n\n', ':\n\n\n')
```

```{code-cell} ipython3
patt = re.compile(
    "(^[A-Z].+?):$"  # the speaker
    "\n{1}(.*?)\n{2}",  # the dialogue
    flags = re.S | re.M
)
```

```{code-cell} ipython3
# Find `patt` such that: 
matches = patt.findall(txt)
df = pd.DataFrame.from_records(matches, columns=['speaker', 'dialogue'])
df.tail()
```

```{code-cell} ipython3
# Create a new column for line number
df['line'] = df.index + 1
df
```

```{code-cell} ipython3
# Split dialogue text into multiple rows
# Ex. split row 7221 into multiple rows based on \n in the dialogue column
(df['dialogue'].str.split('\n', expand=True).stack()
               .reset_index(level=1, drop=True).rename('dialogue'))
```

```{code-cell} ipython3
# Join the splitted dialogue with the original dataframe
df = (df.drop('dialogue', axis=1)
        .join(df['dialogue'].str.split('\n', expand=True).stack()
              .reset_index(level=1, drop=True).rename('dialogue'))
        .reset_index(drop=True))
df
```

## Part 2

You have likely noticed that the lines are not all from the same play!
Now, we will add some useful metadata to our table: 

- Determine a likely source title for each line
- add the title as a 'play' column in the data table. 
- make sure to document your decisions, assumptions, external data sources, etc. 

This is fairly open-ended, and you are not being judged completely on _accuracy_. 
Instead, think outside the box a bit as to how you might accomplish this, and attempt to justify whatever approximations or assumptions you felt were appropriate.

+++

I found a Kaggle dataset about Shakespeare plays (https://www.kaggle.com/kingburrito666/shakespeare-plays). It has the following columns:
1. Dataline
2. Play
3. PlayerLinenumber
4. ActSceneLine
5. Player
6. PlayerLine

```{code-cell} ipython3
# Read in the data
shakespeare = pd.read_csv('Shakespeare_data.csv')
shakespeare[92333:92338]
```

I can determine the play by matching speaker to Player and dialogue to PlayerLine. For easier matches, I will drop the empty dialogues and transform the remaining to plain lowercase text.

```{code-cell} ipython3
# df: drop the empty dialogues
df = df[df['dialogue'] != ''].reset_index(drop=True)
```

```{code-cell} ipython3
# Remove punctuation and lowercase (example)
(shakespeare[92333:92338]['PlayerLine']
 .str.replace(r'[^\w\s]', '', regex=True)
 .str.lower())
```

```{code-cell} ipython3
# df: create a new column for transformed dialogue
df['mat'] = (df['dialogue']
             .str.replace(r'[^\w\s]', '', regex=True)
             .str.lower())
df.tail(3)
```

```{code-cell} ipython3
# shakespeare: create a new column for transformed PlayerLine
shakespeare['che'] = (shakespeare['PlayerLine']
                      .str.replace(r'[^\w\s]', '', regex=True)
                      .str.lower())
shakespeare[92335:92338]
```

```{code-cell} ipython3
# Merge df and shakespeare
# ['speaker','mat'] == ['Player','che']
df = (df
      .merge(shakespeare[['Play', 'Player', 'che']], 
             how = 'left', 
             left_on = ['speaker','mat'], 
             right_on = ['Player','che'])
      .drop(['Player','mat','che'], axis = 1)
      .rename(str.lower, axis='columns'))
df
```

```{code-cell} ipython3
# Check the unmatched dialogues
df[df['play'].isnull()]
```

The number of rows increased from 25554 to 25638, indicating that some dialogues got matched more than once. In addition, there are 64 unmatched dialogues.

+++

**Assumptions:**
- If an unmatch and a match belong to the same line, they should correspond to the same play.
- If the dialogue before an unmatche has a corresponding play and the dialogue after belong to the same play, the unmatch should come from the same play.

```{code-cell} ipython3
# Example of the first assumption
# line 455 belong to Coriolanus
# row 1569 and 1570 should come from the same play
df[1568:1573]
```

```{code-cell} ipython3
for i in range(1,len(df)):
    if df.loc[i,'play'] is np.nan:
        if ((df.loc[i,'line'] == df.loc[i-1,'line']) and # i and i-1 have the same line number
            (df.loc[i-1,'play'] is not np.nan)): # i-1 belongs to a play
            df.loc[i,'play'] = df.loc[i-1,'play'] # assign i with the same play
```

```{code-cell} ipython3
# Example of the second assumption
# row 2746 and row 2748 belong to the same play
# => row 2747 should come from the same play
df[2745:2750]
```

```{code-cell} ipython3
for i in range(1,len(df)-1):
    if df.loc[i,'play'] is np.nan:
        if ((df.loc[i-1,'play'] is not np.nan) and # i-1 belongs to a play
            (df.loc[i-1,'play'] == df.loc[i+1,'play'])): # i+1 belongs to the same play
            df.loc[i,'play'] = df.loc[i-1,'play'] # assign i with the same play
```

```{code-cell} ipython3
# Check the unmatched dialogues now
df[df['play'].isnull()]
```

```{code-cell} ipython3
# Manually find the play for the 4 unmatched dialogues
df = df.fillna('Coriolanus')
df[1823:1827]
```

```{code-cell} ipython3
# Resulting dataframe
df
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
# Create a function for generating word clouds
import matplotlib.pyplot as plt
def value_ct_wordcloud(s: pd.Series):
    from wordcloud import WordCloud
    wc = (WordCloud(background_color="white", max_words=50)
          .generate_from_frequencies(s.to_dict()))
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
```

```{code-cell} ipython3
# Visualize types of play
(df.play.explode().value_counts()
 .pipe(value_ct_wordcloud))
```

```{code-cell} ipython3
# Visualize types of speaker
(df.speaker.explode().value_counts()
 .pipe(value_ct_wordcloud))
```

```{code-cell} ipython3
# Tokenize dialogue
from nltk.tokenize import word_tokenize
df["tokens"] = (df['dialogue']
                .str.replace(r'[^\w\s]', '', regex=True)  # remove punctuation
                .str.lower()  # lower case
                .apply(word_tokenize))
df.head(3)
```

```{code-cell} ipython3
# Remove stopwords from tokens
from nltk.corpus import stopwords
sw = stopwords.words("english")
df['tokens'] = df['tokens'].apply(lambda x: [tkn for tkn in x if tkn not in sw])
df.head(3)
```

```{code-cell} ipython3
# Count the number of types
df.tokens.explode().value_counts().head(30)
```

```{code-cell} ipython3
# Add more stopwords
extra = ['thou','thy','shall','thee','would','hath',
         'let','one','may','us','upon','yet','tis']
sw = sw + extra
```

```{code-cell} ipython3
# Remove extra stopwords from tokens
df['tokens'] = df['tokens'].apply(lambda x: [tkn for tkn in x if tkn not in sw])
df.tail(3)
```

```{code-cell} ipython3
# Generate word clouds for the 10 most frequent speakers
speakers = df.speaker.explode().value_counts().head(10).index.tolist()
df.speaker.explode().value_counts().head(10)
```

```{code-cell} ipython3
speakers[0]
```

```{code-cell} ipython3
# Generate word cloud for a speaker
def speaker_wordcloud(name):
    return (df[df.speaker == name]
            .tokens.explode().value_counts()
            .pipe(value_ct_wordcloud))
```

```{code-cell} ipython3
# Visualize keywords of a speaker
for name in speakers:
    print(name)
    speaker_wordcloud(name)
```

The word clouds above explore how interesting the speakers are. Juliet, for example, is all about love and Romeo, while Queen Margaret mentions a ton about the King and Edward. One weakness of this method is that some keywords do not necessarily represent meanings, but rather the name of the person talking to the speaker.
