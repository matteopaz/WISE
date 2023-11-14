import numpy as np
import pandas as pd

class LightSource: # All the data on a single source from WISE, nicely formatted
    def __init__(self, dataframe):
        # --------------- Class Init --------------- #
        if type(dataframe) != pd.DataFrame:
            raise TypeError("Input must be a pandas dataframe")
        self.pandas = dataframe
        self.numpy = self.pandas.to_numpy()

        # --------------- Datatable Init --------------- #
        tbl = self.numpy

        # Get correct columns
        colnames = self.pandas.columns.tolist()
        mjdindx = colnames.index("mjd")
        w1indx = colnames.index("w1mpro")
        w2indx = colnames.index("w2mpro")
        w1sindx = colnames.index("w1sigmpro")
        w2sindx = colnames.index("w2sigmpro")


        # Type processing and filtering
        dataframe[dataframe == 'null'] = 0 # set "null" strings from WISE catalog to 0
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

        self.datatable = {
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

    def to_tensor(self):
        # Len(pts) x 3 matrix
        w1 = flux["norm"]["w1"]
        w2 = flux["norm"]["w2"]
        dt = flux["norm"]["dt"]
        day = flux["norm"]["day"]
        w1f = flux["norm"]["w1flux"]
        std_val = (flux["norm"]["w1std"] + flux["norm"]["w2std"]) / 2
        std = np.array([std_val for _ in w1])

        # Len(pts) x 3 matrix
        # IMPORTANT! Defines order of data
        return np.stack((w1f, std, day), axis=0).T
    
    def __getitem__(self, key):
        return self.datatable[key]


def get_flux(tbl, colnames):
    # Get correct columns
    mjdindx = colnames.index("mjd")
    w1indx = colnames.index("w1mpro")
    w2indx = colnames.index("w2mpro")
    w1sindx = colnames.index("w1sigmpro")
    w2sindx = colnames.index("w2sigmpro")

    # Type processing
    tbl[tbl == 'null'] = 0
    mjds = tbl[:,mjdindx].astype(float)
    w1mpro = np.nan_to_num(tbl[:, w1indx].astype(float))
    w1sig = np.nan_to_num(tbl[:, w1sindx].astype(float))
    w2mpro = np.nan_to_num(tbl[:, w2indx].astype(float))
    w2sig = np.nan_to_num(tbl[:, w2sindx].astype(float))

    # Sort by date
    sorter = np.argsort(mjds)

    # print(mjds, sorter)

    mjds = mjds[sorter]
    w1mpro = w1mpro[sorter]
    w1sig = w1sig[sorter]
    w2mpro = w2mpro[sorter]
    w2sig = w2sig[sorter]

    # Analysis

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
    w1norm = []
    w2norm = []
    for mag in w1mpro:
        w1norm.append((mag - w1mean) / (w1std)) # Z-score divided by 5.

    for mag in w2mpro:
        w2norm.append((mag - w2mean) / (w2std))

    w1norm = np.nan_to_num(np.array(w1norm))
    w2norm = np.nan_to_num(np.array(w2norm))


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
    day_norm = day / np.max(day)

    # Times since last observation

    dt = [day[i+1] - day[i] for i in range(len(day) - 1)]
    dt = [np.median(dt)] + dt


    # print("day", day)
    # print("dt", dt)


    # Normalized times since last observation

    dt_norm = np.array([np.arcsinh(dt_ex / np.median(dt)) for dt_ex in dt]).flatten()




    data_dict = {
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
        "analyze": {
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


    return data_dict