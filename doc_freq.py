import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")
from nltk import word_tokenize
from collections import defaultdict
import math
import pdb

# prefix = "/media/alex/Windows/git/iati_ag/"
prefix = "/git/iati_ag/"

df = pd.read_csv(prefix+"all.csv",header=0,encoding="latin1")

def tokenize(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    return [w for w in words if w not in cachedStopWords and not w.isdigit() and w not in punctuation]

vocabulary = set()
for description in df["description"].values.tolist():
    words = tokenize(description)
    vocabulary.update(words)
    
vocabulary = list(vocabulary)
word_index = {w: idx for idx, w in enumerate(vocabulary)}
VOCABULARY_SIZE = len(vocabulary)
DOCUMENTS_COUNT = len(df["description"].values)

word_idf = defaultdict(lambda: 0)
for description in df["description"].values.tolist():
    words = set(tokenize(description))
    for word in words:
        word_idf[word] += 1
 
for word in vocabulary:
    word_idf[word] = math.log(DOCUMENTS_COUNT / float(1 + word_idf[word]))
 
idf = pd.DataFrame(word_idf.items(),columns=['word','idf'])
idf.to_csv(prefix+"idf.csv",index=False,encoding="latin1")