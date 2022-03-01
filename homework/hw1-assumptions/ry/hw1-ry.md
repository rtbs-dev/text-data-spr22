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



The assumptions I made about this context are the followings:
1. Ignore the '\n' appeared in the continuous line when defining line number


```python

import dvc.api as dvc

from pathlib import Path
import pandas as pd
from IPython.display import Code, HTML

import re

import hvplot.pandas
import seaborn as sns
import pandera as pa


```










```python
from dvc.api import read,get_url
import pandas as pd

txt = read('resources/data/shakespeare/shakespeare.txt', 
           repo='https://github.com/TLP-COI/text-data-course')

print(txt[:250])

```

    First Citizen:
    Before we proceed any further, hear me speak.
    
    All:
    Speak, speak.
    
    First Citizen:
    You are all resolved rather to die than to famish?
    
    All:
    Resolved. resolved.
    
    First Citizen:
    First, you know Caius Marcius is chief enemy to the people.
    


def shakespearLine(text):
    '''
    The function is used to translate plain txt file from Shakespear's play into dataframes
    Input: plain txt files
    Output: data frame with attributes:
        "Speaker","Line_num", and "Line"
    '''
    
    s = [] #the list of lines spoken in turns by speakers
    n= [] #the list of names
    l = []#list of line number 
    init =1
    paragraph  = re.split(r'\n\n',text)#find each dialog #seperate each dialog
    for dialog in paragraph:
        name = re.split(':',dialog,1)[0]#find name
        content = re.split(':',dialog,1)[-1] #find content, using [-1] here because the first would be " "
        #print(content)
        sentence = re.split('\n',content)[1:]#split each line in the dialog
        #print(sentence)
        for i in sentence:
            if bool(re.match(".+[\w]+.+", i))== False:
                sentence.remove(i)
        n.append(name)
        s.append(sentence)
        l.append(init)
        init = init+1
        
    name_list = []
    line_list = []
    for i in range(len(n)):
        for sentence in s[i]:
            name_list.append(n[i]) #append the name list according to the number of lines by each speaker
            line_list.append(l[i])
    sentence_list = [ item for elem in s for item in elem] #make nested lists to flat list
    
    #create dataframe using the two lists
    shakespear = pd.DataFrame(sentence_list,name_list).reset_index().rename(columns={"index": "Speaker", 0: "Line"})
    shakespear['Line_num'] = line_list#add line number
    
    #rearrange order
    shakespear = shakespear[['Speaker','Line_num','Line']]
    
    
    return shakespear



```python
def shakespearLine(text):
    '''
    The function is used to translate plain txt file from Shakespear's play into dataframes
    Input: plain txt files
    Output: data frame with attributes:
        "Speaker","Line_num", and "Line"
    '''
    
    s = [] #the list of lines spoken in turns by speakers
    n= [] #the list of names

    paragraph  = re.split(r'\n\n',txt)#find each dialog #seperate each dialog
    for dialog in paragraph:
        name = re.split(':',dialog,1)[0]#find name
        content = re.split(':',dialog,1)[-1] #find content, using [-1] here because the first would be " "
        #print(content)
        sentence = re.split('\n',content)[1:]#split each line in the dialog
        #print(sentence)
        if len(sentence)==0: #notice there are blank verse in the play, substitute the empty list with " "
            sentence = [" "]
        n.append(name)
        s.append(sentence)

    l = list(range(len(paragraph)+1))[1:]
    name_list = []
    line_list = []
    for i in range(len(n)):

        for sentence in s[i]:
            name_list.append(n[i]) #append the name list according to the number of lines by each speaker
            line_list.append(l[i])
    sentence_list = [ item for elem in s for item in elem] #make nested lists to flat list

    #create dataframe using the two lists
    shakespear = pd.DataFrame(sentence_list,name_list).reset_index().rename(columns={"index": "Speaker", 0: "Line"})
    shakespear['Line_num'] = line_list#add line number

    #rearrange order
    shakespear = shakespear[['Speaker','Line_num','Line']]
    
    
    return shakespear

```

s = [] #the list of lines spoken in turns by speakers
n= [] #the list of names

paragraph  = re.split(r'\n\n',txt)#find each dialog #seperate each dialog
for dialog in paragraph:
    name = re.split(':',dialog,1)[0]#find name
    content = re.split(':',dialog,1)[-1] #find content, using [-1] here because the first would be " "
    #print(content)
    sentence = re.split('\n',content)[1:]#split each line in the dialog
    #print(sentence)
    if len(sentence)==0: #notice there are blank verse in the play, substitute the empty list with " "
        sentence = [" "]
    n.append(name)
    s.append(sentence)

l = list(range(len(paragraph)+1))[1:]
name_list = []
line_list = []
for i in range(len(n)):
    
    for sentence in s[i]:
        name_list.append(n[i]) #append the name list according to the number of lines by each speaker
        line_list.append(l[i])
sentence_list = [ item for elem in s for item in elem] #make nested lists to flat list

#create dataframe using the two lists
shakespear = pd.DataFrame(sentence_list,name_list).reset_index().rename(columns={"index": "Speaker", 0: "Line"})
shakespear['Line_num'] = line_list#add line number

#rearrange order
shakespear = shakespear[['Speaker','Line_num','Line']]


```python
df = shakespearLine(txt)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Speaker</th>
      <th>Line_num</th>
      <th>Line</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>First Citizen</td>
      <td>1</td>
      <td>Before we proceed any further, hear me speak.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>All</td>
      <td>2</td>
      <td>Speak, speak.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First Citizen</td>
      <td>3</td>
      <td>You are all resolved rather to die than to fam...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>All</td>
      <td>4</td>
      <td>Resolved. resolved.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>First Citizen</td>
      <td>5</td>
      <td>First, you know Caius Marcius is chief enemy t...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25676</th>
      <td>SEBASTIAN</td>
      <td>7221</td>
      <td>And yet so fast asleep.</td>
    </tr>
    <tr>
      <th>25677</th>
      <td>ANTONIO</td>
      <td>7222</td>
      <td>Noble Sebastian,</td>
    </tr>
    <tr>
      <th>25678</th>
      <td>ANTONIO</td>
      <td>7222</td>
      <td>Thou let'st thy fortune sleep--die, rather; wi...</td>
    </tr>
    <tr>
      <th>25679</th>
      <td>ANTONIO</td>
      <td>7222</td>
      <td>Whiles thou art waking.</td>
    </tr>
    <tr>
      <th>25680</th>
      <td>ANTONIO</td>
      <td>7222</td>
      <td></td>
    </tr>
  </tbody>
</table>
<p>25681 rows × 3 columns</p>
</div>



+++

## Part 2

You have likely noticed that the lines are not all from the same play!
Now, we will add some useful metadata to our table: 

- Determine a likely source title for each line
- add the title as a 'play' column in the data table. 
- make sure to document your decisions, assumptions, external data sources, etc. 

This is fairly open-ended, and you are not being judged completely on _accuracy_. 
Instead, think outside the box a bit as to how you might accomplish this, and attempt to justify whatever approximations or assumptions you felt were appropriate. 



By looking at the data, I noticed that the lines speak by one speaker have similar line number. Therefore I assume that the plays are displayed as a whole context instead of dividing into parts. 


```python
plays = pd.read_csv("Shakespeare_data.csv")
```


```python
plays.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataline</th>
      <th>Play</th>
      <th>PlayerLinenumber</th>
      <th>ActSceneLine</th>
      <th>Player</th>
      <th>PlayerLine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Henry IV</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ACT I</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Henry IV</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>SCENE I. London. The palace.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Henry IV</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Enter KING HENRY, LORD JOHN OF LANCASTER, the ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Henry IV</td>
      <td>1.0</td>
      <td>1.1.1</td>
      <td>KING HENRY IV</td>
      <td>So shaken as we are, so wan with care,</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Henry IV</td>
      <td>1.0</td>
      <td>1.1.2</td>
      <td>KING HENRY IV</td>
      <td>Find we a time for frighted peace to pant,</td>
    </tr>
  </tbody>
</table>
</div>




```python
player = plays[['Play','Player','PlayerLine']]
```


```python
player.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Play</th>
      <th>Player</th>
      <th>PlayerLine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Henry IV</td>
      <td>NaN</td>
      <td>ACT I</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Henry IV</td>
      <td>NaN</td>
      <td>SCENE I. London. The palace.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Henry IV</td>
      <td>NaN</td>
      <td>Enter KING HENRY, LORD JOHN OF LANCASTER, the ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Henry IV</td>
      <td>KING HENRY IV</td>
      <td>So shaken as we are, so wan with care,</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Henry IV</td>
      <td>KING HENRY IV</td>
      <td>Find we a time for frighted peace to pant,</td>
    </tr>
  </tbody>
</table>
</div>




```python
title = df.merge(player, how = 'left',left_on=['Line','Speaker'], right_on=['PlayerLine','Player'])
```


```python
title.isna().sum()
```




    Speaker          0
    Line_num         0
    Line             0
    Play          3731
    Player        3731
    PlayerLine    3731
    dtype: int64



At here I assumed that lines which we could not match to a play title was belong to the nearest line with a title. Considering the case of blank verse, assign it to the play which the previous line belongs to.


```python
for i in range(1,len(title)-1):

    m=1
    if title['Play'][i]!=title['Play'][i]:
        #print(i)
        while title['Play'][i+m]!=title['Play'][i+m] and title['Play'][i+m]!=title['Play'][i+m]:
            m = m+1
        
        if title['Play'][i+m]!=title['Play'][i+m]:
            play = title['Play'][i-m]
        else:
            play = title['Play'][i+m]
        title['Play'][i] = play
           

```

    /var/folders/l5/590_1c1955sg4n09g9dcjdbc0000gn/T/ipykernel_28549/4004356965.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      title['Play'][i] = play



```python
title = title[['Speaker','Line_num','Line','Play']]
```


```python
title
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Speaker</th>
      <th>Line_num</th>
      <th>Line</th>
      <th>Play</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>First Citizen</td>
      <td>1</td>
      <td>Before we proceed any further, hear me speak.</td>
      <td>Coriolanus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>All</td>
      <td>2</td>
      <td>Speak, speak.</td>
      <td>Coriolanus</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First Citizen</td>
      <td>3</td>
      <td>You are all resolved rather to die than to fam...</td>
      <td>Coriolanus</td>
    </tr>
    <tr>
      <th>3</th>
      <td>All</td>
      <td>4</td>
      <td>Resolved. resolved.</td>
      <td>Coriolanus</td>
    </tr>
    <tr>
      <th>4</th>
      <td>First Citizen</td>
      <td>5</td>
      <td>First, you know Caius Marcius is chief enemy t...</td>
      <td>Coriolanus</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25727</th>
      <td>SEBASTIAN</td>
      <td>7221</td>
      <td>And yet so fast asleep.</td>
      <td>The Tempest</td>
    </tr>
    <tr>
      <th>25728</th>
      <td>ANTONIO</td>
      <td>7222</td>
      <td>Noble Sebastian,</td>
      <td>The Tempest</td>
    </tr>
    <tr>
      <th>25729</th>
      <td>ANTONIO</td>
      <td>7222</td>
      <td>Thou let'st thy fortune sleep--die, rather; wi...</td>
      <td>The Tempest</td>
    </tr>
    <tr>
      <th>25730</th>
      <td>ANTONIO</td>
      <td>7222</td>
      <td>Whiles thou art waking.</td>
      <td>The Tempest</td>
    </tr>
    <tr>
      <th>25731</th>
      <td>ANTONIO</td>
      <td>7222</td>
      <td></td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>25732 rows × 4 columns</p>
</div>




```python
len(title['Line_num'].unique())
```




    7222



Notice that there are cases in which people with same name speak same line in different plays, which leads to further duplicate. Remove these manually


```python
df3 = title.groupby(['Line_num','Speaker','Play'])['Line'].apply(' '.join).reset_index()
```


```python
df3.loc[df3['Line_num'].duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Line_num</th>
      <th>Speaker</th>
      <th>Play</th>
      <th>Line</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>329</th>
      <td>329</td>
      <td>BRUTUS</td>
      <td>Julius Caesar</td>
      <td>What's the matter?</td>
    </tr>
    <tr>
      <th>5234</th>
      <td>5233</td>
      <td>CLAUDIO</td>
      <td>Much Ado about nothing</td>
      <td>No.</td>
    </tr>
  </tbody>
</table>
</div>



Line num 5233 and 329 are duplicated


```python
title1 = title.drop([19598,1109])
```

+++

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




```python
## Using tf-idf and BoW
```


```python
len(df3['Play'].unique())
```




    11



test


```python

```
