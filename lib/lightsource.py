import numpy as np
import pandas as pd
import torch

class LightSource: # All the data on a single source from WISE, nicely formatted
    def __init__(self, dataframe):
        # --------------- Class Init --------------- #
        if type(dataframe) != pd.DataFrame:
            raise TypeError("Input must be a pandas dataframe")
        self.pandas = dataframe
        self.pandas.index = range(len(self.pandas))
        self.numpy = self.pandas.to_numpy()
        self.datatable = self.init_datatable()
 
    
    def init_datatable(self):
        tbl = self.numpy

        # Get correct columns
        colnames = self.pandas.columns.tolist()
        mjdindx = colnames.index("mjd")
        w1indx = colnames.index("w1mpro")
        w2indx = colnames.index("w2mpro")
        w1sindx = colnames.index("w1sigmpro")
        w2sindx = colnames.index("w2sigmpro")


        # Type processing and filtering
        # tbl[tbl == 'null' or tbl == pd.NA] = 0 # set "null" strings from WISE catalog to 0
        mjds = tbl[:,mjdindx].astype(float)
        w1mpro = np.nan_to_num(tbl[:, w1indx].astype(float))
        w1sig = np.nan_to_num(tbl[:, w1sindx].astype(float))
        w2mpro = np.nan_to_num(tbl[:, w2indx].astype(float))
        w2sig = np.nan_to_num(tbl[:, w2sindx].astype(float))

        # Sorting by date
        sorter = np.argsort(mjds)
        mjds = mjds[sorter]
        w1mpro = w1mpro[sorter]
        w1sig = w1sig[sorter]
        w2mpro = w2mpro[sorter]
        w2sig = w2sig[sorter]

        
        # Statistics
        w1mean = np.nanmean(w1mpro)
        w1median = np.nanmedian(w1mpro)
        w2mean = np.nanmean(w2mpro)
        w2median = np.nanmedian(w2mpro)
        w1var = np.nanvar(w1mpro)
        w2var = np.nanvar(w2mpro)
        w1std = np.sqrt(w1var)
        w2std = np.sqrt(w2var)
        w1mad = np.nanmedian([abs(mag - w1median) for mag in w1mpro]) # Mean Absolute Deviation - Not used
        w2mad = np.nanmedian([abs(mag - w2median) for mag in w2mpro])

        # Normalize with modified z-scoring
        w1norm = np.array([(mag - w1mean) / w1std for mag in w1mpro])
        w2norm = np.array([(mag - w2mean) / w2std for mag in w2mpro])

        if np.isnan(w1norm).any():
            raise Exception("w1norm has nan")
        if np.isnan(w2norm).any():
            raise Exception("w2norm has nan")
        
        # Optional Flux format
        to_flux_w1 = lambda m: 309.54 * 10**(-m / 2.5)
        to_flux_w2 = lambda m: 171.787 * 10**(-m / 2.5)
        w1flux = to_flux_w1(w1mpro)
        w2flux = to_flux_w2(w2mpro)
        
        # Flux normalization arcsin params
        ADJ = 0
        DIV = 0.001
        w1flux_norm = np.arcsinh((to_flux_w1(w1mpro) - ADJ) /DIV)
        w2flux_norm = np.arcsinh((to_flux_w2(w2mpro) - ADJ) /DIV)

        # Days since first timepoint
        day = mjds - mjds[0]
        day_norm = day / np.max(day) # Normalized days from 0 to 1

        # Times since last observation
        dt = [day[i+1] - day[i] for i in range(len(day) - 1)] # has values for all but first day
        dt = [np.median(dt)] + dt # Making assumption that first day will have a "normal" dt
        # Normalized dt
        dt_norm = np.array([np.arcsinh(dt_ex / np.median(dt)) for dt_ex in dt]).flatten()

        return {
            "raw": {
                "w1": w1mpro,
                "w1flux": w1flux,
                "w1sig": w1sig,
                "w2": w2mpro,
                "w2flux": w2flux,
                "w2sig": w2sig,
                "mjd": mjds,
                "day": day,
                "dt": dt
            },
            "norm": {
                "w1": w1norm,
                "w1flux": w1flux_norm,
                "w1std": w1std,
                "w1sig": w1sig,
                "w2": w2norm,
                "w2std": w2std,
                "w2flux": w2flux_norm,
                "w2sig": w2sig,
                "mjd": mjds,
                "day": day_norm,
                "dt": dt_norm
            },
            "statistics": {
                "mean": {
                    "w1": w1mean,
                    "w2": w2mean
                },
                "median": {
                    "w1": w1median,
                    "w2": w2median
                }
            }
        }
        

    def get_numpy(self):
        return self.numpy

    def get_pandas(self):
        return self.pandas
    
    def get_datatable(self):
        return self.datatable
    
    def get_subset(self, start_idx, end_idx):
        # take only rows from start_idx to end_idx
        trimmed = self.pandas.iloc[start_idx:end_idx]
        return LightSource(trimmed)

    def to_tensor(self):
        # Len(pts) x 3 matrix
        w1 = self.datatable["norm"]["w1"]
        w2 = self.datatable["norm"]["w2"]
        dt = self.datatable["norm"]["dt"]
        day = self.datatable["norm"]["day"]
        w1f = self.datatable["norm"]["w1flux"]
        std_val = (self.datatable["norm"]["w1std"] + self.datatable["norm"]["w2std"]) / 2
        std = np.array([std_val for _ in w1])

        # Len(pts) x 3 matrix
        # IMPORTANT! Defines order of data
        return torch.tensor(np.stack((w1, std, day), axis=0).T)
    
    def __getitem__(self, key):
        return self.datatable[key]

    def __len__(self):
        return len(self.datatable)
