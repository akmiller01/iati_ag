import pandas as pd
import pdb
import random


prefix = "/media/alex/Windows/git/iati_ag/"

#Read in the original DC dataset
df = pd.read_csv(prefix+"to_classify.csv",header=0,encoding="latin1")
keys = pd.read_csv(prefix+"keys.csv",header=0)
keyDict = dict(zip(list(keys.key),list(keys.value)))

#Read in the prediction vectors
pred = pd.read_csv(prefix+"prediction_vectors.csv",header=None)
# pred.columns = keys.value
pred.columns = keys.key

#We never scrambled them, so we can keep them in the same order
dc = df.join(pred)
# dc['max'] = dc[keys.value].idxmax(axis=1)
dc['max'] = dc[keys.key].idxmax(axis=1)

# Write csv
dc.to_csv(prefix+"predictions.csv",index=False,encoding="latin1")

