from sklearn.cluster import DBSCAN
from collections import defaultdict
import numpy as np
import torch
import pandas as pd


def get_labels(dataframe):
    dataframe = dataframe.fillna(0.0)
    
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
    


def tbl_to_torch(dataframe: pd.DataFrame): # Based off of LightSource
    w1 = dataframe["w1mpro"].values
    mjd = dataframe["mjd"].values

    to_flux_w1 = lambda x: 309.54 * 10**(-x/2.5) # Jy
    w1 = to_flux_w1(w1)

    # reject 4-sigma outliers
    idxr = w1 - np.mean(w1) < 4 * np.std(w1)
    w1 = w1[idxr]
    mjd = mjd[idxr]
    
    # Extract "snr" estimate from abs flux
    snr_est = -np.log10(w1) / 3
    # zscore w1flux
    w1 = (w1 - np.mean(w1)) / np.std(w1)
    # apply arcsinh
    w1 = np.arcsinh(w1)

    # fix mjd
    mjd = mjd - np.min(mjd)
    mjd = mjd / 4000

    # sort
    idx = np.argsort(mjd)
    mjd = mjd[idx]
    w1 = w1[idx]
    snr_est = snr_est[idx]

    arr = np.array([w1, snr_est, mjd]).T
    return torch.tensor(arr).to(torch.float32).cuda()


class BatchGen:
    def __init__(self, dataframe, labelmap, batchsize):
        self.df = dataframe
        self.lbmap = labelmap
        self.lbs = list(labelmap.keys())
        self.bs = batchsize
        self.current_idx = 0
        self.maxidx = len(self.lbs)
        # ADD SORT BY SIZE
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.current_idx >= self.maxidx:
            raise StopIteration()
        
        centers = []
        stars = []
        for _ in range(self.bs):
            if self.current_idx < self.maxidx:
                labelmap = self.lbmap[self.lbs[self.current_idx]]
                tbl = self.df.iloc[labelmap]
                center = tbl[["ra", "dec"]].mean().values
                self.current_idx += 1
                star = tbl_to_torch(tbl)
                centers.append(center)
                stars.append(star)
        batched = torch.nn.utils.rnn.pad_sequence(stars, batch_first=True).cuda() # pads with 0s
        return (centers, batched)
            
# class BatchGenClustered:
#     def __init__(self, dataframe, batchsize):
#         self.df = dataframe
#         self.bs = batchsize
#         self.current_idx = 0
#         self.maxidx = len(self.lbs)
#         # ADD SORT BY SIZE
        
#     def __iter__(self):
#         return self
        
#     def __next__(self):
#         if self.current_idx >= self.maxidx:
#             raise StopIteration()
        
#         centers = []
#         stars = []
#         for _ in range(self.bs):
#             if self.current_idx < self.maxidx:
#                 labelmap = self.lbmap[self.lbs[self.current_idx]]
#                 tbl = self.df.iloc[labelmap]
#                 center = tbl[["ra", "dec"]].mean().values
#                 self.current_idx += 1
#                 star = tbl_to_torch(tbl)
#                 centers.append(center)
#                 stars.append(star)
#         batched = torch.nn.utils.rnn.pad_sequence(stars, batch_first=True).cuda() # pads with 0s
#         return (centers, batched)            
