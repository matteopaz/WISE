from sklearn.cluster import DBSCAN
from collections import defaultdict
import numpy as np
from datetime import datetime
from joblib import parallel_backend
import pickle
import pandas as pd
import time

csv = "orionsbelt25deg"

print("Reading CSV")
dataframe = pd.read_csv("./runner/inp/{}.csv".format(csv))
dataframe = dataframe.fillna(0.0)
print("CSV Read")

t1 = time.time()

print("Clustering.. started at", datetime.now().strftime("%H:%M:%S"))
def cluster(dataframe):
    clusters = defaultdict(list)

    eps = 1/3600 # 1 arcsec
    min_samp = 10
    pos = dataframe[["ra", "dec"]]

    print(len(pos), "rows")
    dbscan = DBSCAN(eps=eps, min_samples=min_samp, n_jobs=-1)
    clustered = dbscan.fit(pos)
    labels = clustered.labels_


    for i, clusternum in enumerate(labels):
        clusters[clusternum].append(i)
    return clusters

c = cluster(dataframe)
t2 = time.time()
print("Time Elapsed: ", t2 - t1)


with open("./runner/cluster/{}_ptrs.pkl".format(csv), "wb") as f:
    pickle.dump(c, f)