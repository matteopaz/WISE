from sklearn.cluster import DBSCAN
from collections import defaultdict
import numpy as np
from datetime import datetime
import pickle
import pandas as pd
import time
import json 

csv = "1deg"

print("Reading CSV")
dataframe = pd.read_csv("./runner/inp/{}_clean.csv".format(csv))
print("CSV Read")

t1 = time.time()

print("Clustering.. started at", datetime.now().strftime("%H:%M:%S"))
def cluster(df):
    clusters = defaultdict(list)
    pos = df[["ra", "dec"]]

    print(len(pos), "rows")
    dbscan = DBSCAN(eps=1/3600, min_samples=16, n_jobs=-1, algorithm="kd_tree")
    labels = dbscan.fit_predict(pos)
    
    for i, clusternum in enumerate(labels):
        if clusternum != -1:
            clusters[clusternum].append(i)
    
    # delete clusters with less than 100 members
    for k in list(clusters.keys()):
        idxs = clusters[k]
        median_w1 = df.iloc[idxs]["w1mpro"].median()
        if len(idxs) < 100 and (median_w1 > 13 or len(idxs) < 20): # keep only well sampled or potential novae
            del clusters[k]

    return clusters

c = cluster(dataframe)
t2 = time.time()
print("Time Elapsed: ", t2 - t1)

with open("./runner/cluster/{}_ptrs.pkl".format(csv), "wb") as f:
    pickle.dump(c, f)