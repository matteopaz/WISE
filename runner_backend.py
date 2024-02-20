from sklearn.cluster import DBSCAN
from collections import defaultdict
import numpy as np
import torch


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
    


def tbl_to_torch(dataframe): # Based off of LightSource
    w1 = dataframe["w1mpro"].to_numpy()
    mjd = dataframe["mjd"].to_numpy()
    
    w1 = np.nan_to_num(w1.astype(float), nan=np.nanmean(w1))
    
    sorter = np.argsort(mjd)
    mjd = mjd[sorter]
    w1 = w1[sorter]
    
    ra = dataframe["ra"].to_numpy()
    dec = dataframe["dec"].to_numpy()
    

    center = (np.mean(ra), np.mean(dec))
    
    
    to_flux_w1 = lambda m: 309.54 * 10**(-m / 2.5)
    w1f = to_flux_w1(w1)

    std = -np.log10(w1f) / 3 # to get NN friendly inp

    w1z = (w1f - np.mean(w1f)) / np.std(w1f)
    idxr = np.where(np.abs(w1z) < 5)[0] # trim outliers

    w1z = w1z[idxr]
    std = std[idxr]
    mjd = mjd[idxr]

    w1 = np.arcsinh(w1z)
    
    day = mjd - mjd[0]
    day = day / np.max(day)
    
    tnsr = torch.tensor(np.stack((w1, std, day), axis=0).T).cuda() # T x F
    return center, tnsr.to(torch.float32)
    
    

class BatchGen:
    def __init__(self, dataframe, labels, batchsize):
        self.df = dataframe
        self.lb = labels
        self.bs = batchsize
        self.current_idx = 0
        self.maxidx = len(self.lb) - 1
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
                tbl = self.df.iloc[self.lb[self.current_idx]]
                self.current_idx += 1
                center, star = tbl_to_torch(tbl)
                centers.append(center)
                stars.append(star)
        batched = torch.nn.utils.rnn.pad_sequence(stars, batch_first=True).cuda() # pads with 0s
        return (centers, batched)
            
            
            
        
         
    