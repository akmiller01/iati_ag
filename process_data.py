import pandas as pd
import numpy
from csv import QUOTE_NONE
from csv import QUOTE_ALL
import pdb
import random
from nltk.corpus import stopwords
# cachedStopWords = stopwords.words("english")+stopwords.words("french")+stopwords.words("norwegian")+stopwords.words("german")+stopwords.words("dutch")
cachedStopWords = stopwords.words("english")

prefix = "/media/alex/Windows/git/iati_ag/"

#Function to remove stop words
def removeStop(x):
    return " ".join([word for word in x.split() if word.lower() not in cachedStopWords])

#Process our NYC data for neural network
df = pd.read_csv(prefix+"iati_ag_preprocess.csv",header=0,encoding="latin1")

#Reorder data
df = df[["sector","description"]]

#Let's remove the return characters, add spacing to common punctuation, and remove stop words
def clean_text(row):
    return row.replace("\r","").replace("\n"," ").replace(","," ").replace("."," . ").replace("/"," / ").replace('"',' " ').replace('!',' ! ').replace('?',' ? ').replace('$',' $ ').replace('-',' - ').replace(")"," ) ").replace("("," ( ")
df["description"] = df["description"].apply(clean_text).apply(removeStop)

#And stringify scores
def stringify(row):
    return str(int(row))
df["sector"] = df["sector"].apply(stringify)

#Create balanced set for training
unique_sector_names = df.sector.unique()
unique_sectors = range(1,len(unique_sector_names)+1)
segmented_descriptions = {}
for sector in unique_sectors:
    segmented_descriptions[sector] = []
    
for index, row in df.iterrows():
    sector_name = row['sector']
    sector_index = numpy.where(unique_sector_names==sector_name)[0][0]+1
    description = row['description']
    segmented_descriptions[sector_index].append(description)

sector_length = min([len(segmented_descriptions[sector]) for sector in unique_sectors])

for sector in unique_sectors:
    random.shuffle(segmented_descriptions[sector])
    segmented_descriptions[sector] = segmented_descriptions[sector][:sector_length]

# Segment the data randomly into training and testing
training_ratio = 0.90
training_count = int(sector_length*training_ratio)

all_training = []
all_testing = []
all_sectors = []

for sector in unique_sectors:
    all_sectors.append(pd.DataFrame({"sector":sector,"description":segmented_descriptions[sector]}))
    all_training.append(pd.DataFrame({"sector":sector,"description":segmented_descriptions[sector]}).ix[:training_count])
    all_testing.append(pd.DataFrame({"sector":sector,"description":segmented_descriptions[sector]}).ix[training_count:])

df = pd.concat(all_sectors,ignore_index=True)
train = pd.concat(all_training,ignore_index=True)
test = pd.concat(all_testing,ignore_index=True)
df = df[["sector","description"]]
train = train[["sector","description"]]
test = test[["sector","description"]]

# Write csvs
train.to_csv(prefix+"train.csv",index=False,encoding="latin1")
test.to_csv(prefix+"test.csv",index=False,encoding="latin1")
df.to_csv(prefix+"all.csv",index=False,encoding="latin1")

keys = pd.DataFrame({"key":unique_sectors,"value":unique_sector_names})
keys.to_csv(prefix+"keys.csv",index=False)