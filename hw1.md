---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Python 3.8.12 ('base')
    language: python
    name: python3
---

## Load Packages

```python
import pandas as pd
import numpy as np
from dvc.api import read, get_url
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib.pyplot import figure
```

## Part 1


Split the text into a table with each row representing a line of dialogue with columns for the speaker, line number, and the actual dialogue

```python
# read in and open the text
txt = open('shakespeare.txt', 'r')
txt = txt.read()
```

```python
# let's see what we're working with here by printing the first few lines
print(txt[:250])
```

```python
# what does it look like not nearly formatted by print?
txt[:250]
```

Based on what we've seen above what assumptions can we make about the text?
- Assumption 1: All new lines start with a capial letters
- Assumption 2: All sections are separated by a double new line
- Assumption 3: A colon followed by a space separates spakers from their dialogue
- Assumption 4: All documents follow the pattern "Speaker: Dialogue"

```python
# save the first bit to test our assumptions since it's small and easy to work with 
test = txt[:250]
```

Our regex will require two capture groups- one for the speaker and the other for dialogue.

I assumed all lines begin with a capital but this is Shakespeare after all and he's prone to do anything (and if not then maybe there might be typos) so we'll allow the pattern to account for lowercase letters at the beginning of speaker names since our last assumption is that it follows the pattern "Speaker: Dialogue"

```python
# put together regex pattern
patt = re.compile(
    "(?:\A|\n\n)"  # beginning of file or two newlines
    "(^[A-Z][\w ]+):$"  # Can start with Upper or lower case and ends with a colon
    "\n([\w\W]+?)"  # ANYTHING, but lazy (slightly modified from professor's code in slack but this is word/not a word instead of whitespace/not whitespace)
    "(?=\n\n|\Z)",  # until you hit two newlines or end-of-file
    flags=re.M
)

test_match = patt.findall(test)
```

```python
# let's see if this works on our test set
test_df = pd.DataFrame(test_match, columns=['Speaker', 'Dialogue'])
test_df
```

It looks like our assumptions hold!

```python
# next let's add the document line number

test_df["Line Number"] = np.arange(len(test_df))

# This starts at 0 so add 1
test_df["Line Number"] = test_df["Line Number"] + 1

# it works! 
test_df
```

```python
# Apply what we've done above to the whole doucument
full_match = patt.findall(txt)
df = pd.DataFrame(full_match, columns=['Speaker', 'Dialogue'])
df["Line Number"] = np.arange(len(df))
df["Line Number"] = df["Line Number"] + 1
df
```

```python
# there are still some random new line characters throughout so let's strip those for good measure
clean_lines = []
for line in df["Dialogue"]:
    clean_lines.append(line.replace("\n", " "))

df["Dialogue"] = clean_lines
df
```

## Part 2


Determine the title of the play and add it as a 'Play' column to the table


My first assumption was that shakespeare wouldn't have have recurring characters so we could match plays by character. However, a quick google search proved this to be false so there goes that easy out. On the bright side, in looking this up  I found a kaggle dataset of shakespeare plays: https://www.kaggle.com/kingburrito666/shakespeare-plays

We can load this and cross reference the dialogue in this dataset with our smaller one

```python
# load the full shakespeare dataset
Big_Shake = pd.read_csv('Shakespeare_data.csv')
Big_Shake
```

```python
# okay this is promising here we have the play lined up with each characters name and line
# first let's keep only what we need
Big_Shake.drop(columns=["Dataline", "ActSceneLine", "PlayerLinenumber"], inplace=True)

# before we join let's try to standardize as mich as we can by removing punctuation and making everything lowercase
# make both dfs lowercase
Big_Shake = Big_Shake.applymap(lambda s: s.lower() if type(s) == str else s)
df = df.applymap(lambda s: s.lower() if type(s) == str else s)

# remove punctuation from both
Big_Shake = Big_Shake.applymap(lambda s: re.sub(r'[^\w\s]', '', s) if type(s) == str else s)
df = df.applymap(lambda s: re.sub(r'[^\w\s]', '', s) if type(s) == str else s)

# view the datasets to make sure everything looks right
Big_Shake
```

```python
df
```

Assumption prior to merge: If a documents matches both speaker and dialogue it must be from the same play

```python
# okay let's try to join on dialogue
merged_df = df.merge(Big_Shake, how = "left", left_on = ["Dialogue", "Speaker"], right_on=["PlayerLine", "Player"])

merged_df

```

```python
# what didn't merge?

merged_df[merged_df['Play'].isna()]

# oh shoot there's a lot- this dind't work nearly as well as I hoped by maybe we can impute some
```

Assumption for imputation: if the speaker is the same as two lines before (so in this case there would be active dialogue) let's assume that they're the same play

```python
# impute
for i, row in merged_df.iterrows():
    if merged_df.loc[i,'Play'] is np.nan:
        if merged_df.loc[i,'Speaker'] == merged_df.loc[i-2,'Speaker']:  
            merged_df.loc[i,'Play'] =  merged_df.loc[i-2,'Play']

# that made a big difference!
merged_df[merged_df['Play'].isna()]
```

```python
# can we do it backwards now? So if the speaker is the same as two lines after, we can assume that it is the same play
for i, row in merged_df.iterrows():
    if merged_df.loc[i,'Play'] is np.nan:
        if merged_df.loc[i,'Speaker'] == merged_df.loc[i+2,'Speaker']:  
            merged_df.loc[i,'Play'] = merged_df.loc[i+2,'Play']

# that made a big difference!
merged_df[merged_df['Play'].isna()]
```

We were able to impute a singificant amount of the missing values. This imputation is based on the assumption that even if characters are reused they wouldn't be reused so frequently as to be within two lines of each other. Instead, if a character's lines are so close together they must be engaged in active dialogue and therefore it is the same play

```python
# Our new dataset is largere than our original, why is that?
# let's see if there are duplicate dialogues

# here are our duplicates
merged_df[merged_df["Line Number"].duplicated(keep=False)]
```

Based on the above duplicates, we can assume that there are some plays where there are repeared characters- such as Coriolanus and Julius Caesar who use similar short lines. These then matched with each other in the merge, creating duplicates. The incomplete matches and duplicates show that while this method kind of worked, it was far from perfect


## Part 3


Determine how important or interesting a speaker is


Assumption 1: The person who speaks the most is the most important

```python
# speaker frequency
counts = df["Speaker"].value_counts()[:10]
counts = pd.DataFrame(counts).reset_index().rename(columns={'index':'Speaker', 'Speaker':'Count'})
```

```python
# let's plot the frequency
plt.figure(figsize=(15, 5))
plt.bar("Speaker", "Count", data = counts)
plt.xlabel("Character")
plt.ylabel("Frequency")
plt.title("Character Lines")
plt.show()


```

I don't know who this gloucester fellow is but after some googling he seems to be a side character at best. This would imply that our frequency idea was not a very good one


New assumption: Maybe it's not how much you talk but what you say. Let's see if we can determine importance by which characters use keywords the most

```python
# list of stopwords I copied off this github comment because it had words like thou: https://gist.github.com/sebleier/554280?permalink_comment_id=3059054#gistcomment-3059054
# I then tweaked it a little more for this dataset
stopwords = ['thee', 'hath', 'should', 'tis', 'man', 'she',"0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", 'his', 'him', 'thy', 'will', 'her', 'good', "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"]
```

```python
# frequency of keywords
# what are the most used words
corpus = df["Dialogue"]
vectorizer = CountVectorizer(stop_words= stopwords)
X = vectorizer.fit_transform(corpus)
dtm = vectorizer.transform(corpus)
dtm = pd.DataFrame(dtm.toarray(), columns = vectorizer.get_feature_names())
# stopwrods list includes things like thou

```

```python
freq = dtm.sum(axis = 0).sort_values(ascending = False)
most_freq = pd.DataFrame(freq[:30]).reset_index().rename(columns={'index':'Word', 0:'Count'})
most_freq = list(most_freq.Word)
```

```python
# who uses those words the most
pattern = '|'.join(most_freq)
most_common = df[df['Dialogue'].str.contains(pattern)]

```

```python
counts_2 = most_common["Speaker"].value_counts()[:10]
counts_2 = pd.DataFrame(counts_2).reset_index().rename(columns={'index':'Speaker', 'Speaker':'Count'})
plt.figure(figsize=(15, 5))
plt.bar("Speaker", "Count", data = counts_2)
plt.xlabel("Character")
plt.ylabel("Frequency")
plt.title("Character Lines w Keywords")
plt.show()

```

Pretty much the same result in a slightly different order. This doesn't seem like it was a great idea either. A better approach would likely account for how a charcter's speech moves the plot, but not quite sure how to do that here



