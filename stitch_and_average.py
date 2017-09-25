import pandas as pd
import pdb
import random


prefix = "/media/alex/Windows/git/iati_ag/"

#Read in the original DC dataset
df = pd.read_csv(prefix+"to_classify.csv",header=0,encoding="latin1")
keys = pd.read_csv(prefix+"keys.csv",header=0)
keyDict = dict(zip(list(keys.key),list(keys.value)))

def penDiff(series):
    return -series.sort_values(inplace=False,ascending=False)[:2].diff().values[1]

#Read in the prediction vectors
pred = pd.read_csv(prefix+"prediction_vectors.csv",header=None)
# pred.columns = keys.value
pred.columns = keys.key
pred_norm = pred.sub(pred.min(axis=1),axis=0)
pred_norm = pred_norm.div(pred_norm.max(axis=1),axis=0)
pred_norm['confidence'] = pred_norm.apply(penDiff,axis=1)
pred_norm['max'] = pred_norm.idxmax(axis=1)

#We never scrambled them, so we can keep them in the same order
dc = df.join(pred_norm)

# Write csv
dc.to_csv(prefix+"predictions.csv",index=False,encoding="latin1")

