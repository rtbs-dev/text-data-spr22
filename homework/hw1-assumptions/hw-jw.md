---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.6
kernelspec:
  display_name: Python [conda env:PS1]
  language: python
  name: conda-env-PS1-py
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
from dvc.api import read
import pandas as pd
import numpy as np
import janitor as pj
import re
import nltk
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

txt = read('shakespeare.txt')

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

+++

### Assumptions

1. All text follows the same format:
- One line with the speaker followed by a colon
- One block of text of dialogue with no breaks within. 
- Two line breaks after the dialogue. 
    
2. Text does not include accents or non-Roman characters

```{code-cell} ipython3
### Prepare dataframe

pattern = re.compile(
        "((?<!(\S\n))^[\w -:]*):\n"
        "([\w -:;,.?!]*\n[\w -:;,.?!]*)"
        ,
        flags = re.S | re.M
        )

matches = pattern.findall(txt)

df = pd.DataFrame.from_records(matches, columns=["speaker", "line", "dialogue"])

## Replace empty column with line number

df['line'] = range(1, len(output) + 1) 

## Replace line breaks in dialogue with spaces

df = df.replace(to_replace ='\n', value = ' ', regex=True) 

## Make speaker names upper-case (for Part 2)

df['speaker'] = df['speaker'].str.upper()

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

### Approach and Assumptions

1. Join my table to a table of characters by play. Several characters will have multiple matches. (source:  https://www.playshakespeare.com/study/complete-shakespeare-character-list)
1. **Assumption**: for lines with no matching play - they are in response to the previous line, and are thus the same play. If the previous line has multiple matching plays, this would mean that the unmatched line would be matched to the last play listed for the previous line, which may not be accurate.
1. For lines with multiple matches, determine best match according to:
    
    - Matching plays with adjacent lines:
    
        - **Assumption** If adjacent lines have only one matching play, then both lines should match with that line. Check first with the previous line, then the following line. 
        
        - **Assumption** If adjacent lines have multiple matching plays, the play with highest combined number of lines for both involved characters will match. This assumption treats dialogues are between two characters, so in theory one dialogue with 3+ characters that all appear in multiple plays could be erroneously matched to two separate plays. 

    - For remaining multi-matched lines:
        - **Assumption** Characters with more lines are more likely to have their dialogue included in this corpus. Match lines to the play where the character has the most lines. 
        - **Bad Assumption** For remaining multi-matched lines, the first play they are matched to is the most likely. Drop all duplicates but first instance of a line. 

+++

## Assumptions 
### List assumptions here.
- IF THIS, THEN THIS
- IF NOT THIS, THEN THIS
- IF NOT THIS, THEN THIS

```{code-cell} ipython3
### Prepare character names in new dataset and join to df
## Source: https://www.playshakespeare.com/study/complete-shakespeare-character-list
## Accessed: February 28, 2022

plays = pd.read_csv('plays.csv')

plays['Character'] = plays['Character'].str.upper()

## Remove character name abbreviations

pattern = r"(?P<speaker>.+)(?= [(|(\d])"
plays_new = plays['Character'].str.extract(pattern, expand=True)

plays = plays.join(plays_new)

plays.drop(['Character'], axis=1, inplace=True)

df = pd.merge(df,
              plays,
              on = 'speaker',
              how = 'left')
```

```{code-cell} ipython3
### Some initial cleaning and matching plays to lines with no match

df['Lines'] = df['Lines'].fillna(0)

df['Lines'] = df['Lines'].astype(int)

last_play = { 'play': None, 'line': None}

## Assume that for missing lines, that line is in response to the previous line

df['Play'].fillna(method='bfill', inplace=True) 
```

```{code-cell} ipython3
### Identifying play for lines with multiple matches

## Determine number of matches per line

df['yes'] = 1

df['sum'] = df.groupby('line')['yes'].transform(len)

## Determine number of matches for preceding or subsequent lines
# We'll use adjacent lines to help identify play
# A multi-match line should have at least one of the same plays as one adjacent line

df1 = df.copy()

df1 = df1.drop_duplicates(subset = 'line',
                          keep = 'last')

df1['summinus'] = df1['sum'].shift(1, axis = 0)

df1['sumplus'] = df1['sum'].shift(-1, axis = 0)

df1 = df1[['line', 
         'summinus', 
         'sumplus']]

df = pd.merge(df,
              df1,
              on = 'line',
              how = 'left')

merge = df[['line','sum']]

merge.drop_duplicates(subset = "line", keep = "first", inplace = True)
```

```{code-cell} ipython3
### Merge in sum data for multi-match and adjacent lines 

# One version with missing values left as missing to identify 
# matching plays between lines in next cell
# Values will become 0/1 rather than number of lines

df2 = df[(df['sum'] > 1) | (df['summinus'] > 1) | (df['sumplus'] > 1)] 

df2 = pd.pivot(df2, index='line', columns='Play', values='yes')

df2 = pd.merge(df2, merge, on="line", how="left")

df2.shape ## 39 columns, 2784 rows

# One version with missing values replaced with zero 
# Values will remain number of lines

df3 = df[(df['sum'] > 1) | (df['summinus'] > 1) | (df['sumplus'] > 1)] 

df3 = pd.pivot(df3, index='line', columns='Play', values='Lines')

df3 = pd.merge(df3, merge, on="line", how="left")

df3.fillna(0, inplace = True)
df3['sum2'] = 1 ## so when multipled by df4 in next cell it doesn't become 0
```

```{code-cell} ipython3
### Determine matching plays between multi-match lines and adjacent

df4 = df2.copy()

# Identify matches based on previous lines. 
# If a line with a play match doesn't match with the previous line, pull in the value (1/0) from the subsequent line. 
for name, values in df4.iloc[:, 1:38].iteritems():
    df4[name] = df4[name].diff().fillna(df4[name]) # -/+1 = no match with previous, NaN = no match, 0 = match
    df4[name] = df4[name].replace(to_replace = 0, value = 10) # change match to 10
    df4[name] = df4[name].replace(to_replace = 1, value = 99) # change no match to 99
    df4[name] = df4[name].replace(to_replace = -1, value = 0) # change no match to 0
    df4[name] = df4[name].replace(to_replace = 10, value = 1) # recode match (10) as 1
    df4[name] = df4[name].fillna(0) # missing => 0
    df4[name].replace(99, np.nan, inplace = True) # 99 => missing
    df4[name].bfill(axis = 0, inplace = True) # replace missing with following line

# Sum of matches across rows 
df4['sum2'] = df4.iloc[:, 2:39].sum(axis=1)
df4['sum2'] = df4['sum2'] - df4['sum']
```

```{code-cell} ipython3
### For plays with multiple neighbor matches, assume play with highest average number of lines across speakers

# Add number of lines for potential matches (stored in d4)
df5 = df3.mul(df4, fill_value = 0, level = "int")

df6 = df5.copy()

# Find difference in line amount for matching values (stored in d6)
for name, values in df6.iloc[:, 2:39].iteritems():
    df6[name] = df6[name].diff().fillna(df6[name])

df6_inv = df5-df6

# Average number of lines 
df7 = (df5 + df6_inv)/2

# Keep play with max average number of lines across neighbors for multiple neighbor matches, else replace with 0
df7['max'] = df7.iloc[:, 1:38].max(axis=1)

df8 = df7.copy()

# Manually input line 58 (first row), which gets lost 
df8.at[0, 'Titus Andronicus'] = 1    

df8['max'] = df8.iloc[:, 1:38].max(axis=1)

for name, values in df8.iloc[:, 2:39].iteritems():
    df8[name] = df8.apply(lambda x: 1 if x[name] == x['max'] else 0, axis = 1)
```

```{code-cell} ipython3
### Prepare to merge back

# Restore line values

df8['line'] = (2*df8['line'])**(1/2)

# Drop unnecessary columns

df8.drop(['sum', 'sum2', 'max'], axis = 1, inplace = True)

df8.drop(df8.columns[[1]], axis = 1, inplace = True)

# Reshape long

df9 = df8.melt(id_vars = 'line', var_name = 'play')

#df9 = df9[(df9['value'] == 1)]
```

```{code-cell} ipython3
### Merge back in

df10 = pd.merge(df, df9, on = "line", how = "left")

# For multi-match plays, drop erroneous plays ('play' should match 'Play' if multi-match)

df10['match'] = df10.apply(lambda x: 1 if (x['play'] == x['Play']) else 0, axis = 1)

df10['rank'] = df10.groupby('line')['match'].transform('max')

df10 = df10[(df10['match'] == df10['rank'])]

# For plays that still have multiple potential matches, keep the play where they have the highest number of lines

df10['duplicate'] = df10['line'].duplicated(keep == False)

df10['line_max'] = df10.groupby('line')['Lines'].transform('max')
 
df11 = df10[((df10['duplicate'] == True) & (df10['Lines'] == df10['line_max'])) | (df10['duplicate'] == False)]

# Where duplicates remain, keep first instance

df11.drop_duplicates(subset = 'line', keep ='first', inplace = True) 
```

```{code-cell} ipython3
### Prepare final dataframe

df11 = df11[['speaker', 'line', 'dialogue', 'Play']]

df11.reset_index(inplace = True)
df11.drop('index', axis = 1, inplace = True)

df = df11

display(df)
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

+++

### Question: What characters are messengers of violence?

**Technique** Keyword frequency
1. Tokenize dialogue
2. Remove stopwords, punctuation, ends of contractions
3. Lemmatize tokens
4. Generate list of violence-related key words using WordNet
5. Classify words as violence-related if they fall within the list of key words
6. Determine frequency of violence-related key words per character
7. Determine percentage of words spoken that are related to violence per character
    

**Justification**
1. Violent acts generally represent pivotal moments in plays. While characters speaking about violence aren't necessarily perpetrators of violence or even proximity to the violent act, discussion of violence may make characters more *likely* to be involved in the violent act, and the act of discussing violent acts can mark critical junctures within the play. 
2. I use keyword frequency to measure the extent to which each character is a messenger of violence. While frequency alone tells us how many violence-related words a character speaks, each character has a different amount of dialogue. If one character speaks 5 violence-related words out of 100 words, while another character speaks 5 violence-related words out of 10 (to give an extreme, and extremely unlikely, example), they should not be seen as comparable messengers of violence. Consequently, messengers of violence are identified based on the percentage of words spoken that relate to violence, so we can understand how *often* a character speaks about violence relative to the rest of their dialogue. 

**Assumptions**
1. The list of violence-related words identified through WordNet is comprehensive of all violence-related words in the corpus.
2. Character dialogue within this corpus reflects their dialogue across all of Shakespeare's plays (i.e. if a character has a lot of violence-related dialogue in this corpus, then they have a lot of violence-related dialogue across all plays.). 
3. Acts of violence, when they occur, are discussed in proximity to the act. If someone is killed, it will be mentioned shortly before or after; if someone is attacked, it will be mentioned, etc. This doesn't effect the classification of characters as messengers of violence, but it does effect the utility of the classification. If people don't talk about violent acts when they occur, it's not particularly useful to identify it when it does occur (or maybe it means that they are primary perpetrators of violence?).

**Limitations**
1. Any discussion of violence that falls outside of my list of words will not be captured -> undercount of violence-related words
2. We don't know word use - for example, "assault" could be used as "assault on the senses," which would not necessarily be a discussion of a violent act. -> overcount of violence-related words
3. We can't be sure of proximity of violence discussion to violent acts. Someone mentioning violence may not necessarily be a marker of a violent act recently or immediately occurring - i.e. my primary messengers of violence may not be *the* primary messenger of violence for a given act. This could especially be an issue if a violent act is a central component of a play - it could be discussed repeatedly throughout the play. 

```{code-cell} ipython3
### Tokenize and exclude stopwords, punctuation, ends of contractions. 

wordlists = (df.dialogue.fillna('')
             .str.lower()
             .apply(nltk.tokenize.word_tokenize))

tidy_df = (df.add_column('word', wordlists))

tidy_df = tidy_df.explode('word').reset_index()

tidy_df.word.value_counts().head()

punctuation = [',','.',';',':','?','!', '\'', '-', '--']

regex = "(['-]\w{1,2}$)" ## contraction ends

drop = tidy_df.word[tidy_df.word.str.match(regex)]

tidy_df = tidy_df.filter_column_isin('word', 
                                     nltk.corpus.stopwords.words('english'),
                                     complement = True)
tidy_df = tidy_df.filter_column_isin('word', 
                                     punctuation, 
                                     complement = True)
tidy_df = tidy_df.filter_column_isin('word', 
                                     drop, 
                                     complement = True)

tidy_df.word.value_counts().head(30)
```

```{code-cell} ipython3
### Lemmatize for comparison for comparison to violence-related words from WordNet

word_data = tidy_df['word']

wnl = WordNetLemmatizer()

lemma_string = ' '.join([wnl.lemmatize(w) for w in word_data])

lemma = lemma_string.split()

lemma_df = tidy_df.copy()

lemma_df['lemma'] = lemma 
```

```{code-cell} ipython3
### Compile list of violence-related words from WordNet

set1 = wn.synset('battle.v.01').lemma_names('eng')
set2 = wn.synset('sword.n.01').lemma_names('eng')
set3 = wn.synset('weapon.n.01').lemma_names('eng')
set4 = wn.synset('kill.v.01').lemma_names('eng')
set5 = wn.synset('slay.v.01').lemma_names('eng')
set6 = wn.synset('stab.v.01').lemma_names('eng')
set7 = wn.synset('fight.v.01').lemma_names('eng')
set8 = wn.synset('poison.v.01').lemma_names('eng')
set9 = wn.synset('attack.v.04').lemma_names('eng')
set10 = wn.synset('duel.n.01').lemma_names('eng')
set11 = wn.synset('hurt.v.05').lemma_names('eng')

violence  = set1 + set2 + set3 + set4 + set5 + set6 + set7 + set8 + set9 + set10 + set11
```

```{code-cell} ipython3
### Identify words in dialogue that fall within list of violence-related words

lemma_df['violence'] = lemma_df['lemma'].isin(violence)
lemma_df['violence'] = lemma_df['violence'].map({True:'Yes', False:'No'})

lemma_df.violence.value_counts()

tab = pd.crosstab(lemma_df.speaker, lemma_df.violence).reset_index()
```

```{code-cell} ipython3
### Prepare for plot: percent of words that are violent, sort, reset index 

tab['percent'] = tab.apply(lambda x: (x['Yes'])/(x['Yes'] + x['No']), axis = 1)

tab = tab.sort_values(by=['percent'], ascending = False)
```

```{code-cell} ipython3
### Plot graph

speaker = tab['speaker'].head(15)
violence = tab['percent'].head(15)

fig, ax = plt.subplots(figsize=(20,15))
ax.invert_yaxis()

bar = ax.barh(speaker, violence, color = 'maroon')
plt.tight_layout()

plot_title = 'Percent of spoken words related to violence'
title_size = 28
ax.set_xlabel('(0.1 = 0.1%)', size = 20)

title = plt.title(plot_title, pad=30, fontsize=title_size)
title.set_position([.27, 1])
```
