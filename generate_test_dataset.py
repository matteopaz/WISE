import numpy as np
import pickle
import astropy.units as u
from astroquery.ipac.irsa import Irsa
import pandas as pd

import os
import sys
ROOT = os.path.join("./")
sys.path.append(ROOT + "lib")

from helpers import *
torch.set_default_dtype(torch.float32)

REQUIRED_COLS = ["ra", "dec", "w1mpro", "w2mpro", "qual_frame", "mjd"]

neowise = 'neowiser_p1bs_psd'
Irsa.ROW_LIMIT = 100000

obj_qual = 3
min_qual_frame = 4

testdb = pd.DataFrame()

key = {}

for roots, dirs, files in os.walk(ROOT + "object_spreadsheets/"):
    for spreadsheet in files:
        kind = spreadsheet[:-4] # remove .csv

        df = pd.read_csv(ROOT + "object_spreadsheets/" + spreadsheet)
        df.dropna(axis=0, how="any", inplace=True)

        i = 0
        for objname, ra, dec, rad, qual in zip(df["Name"], df["RAJ2000"], df["DecJ2000"], df["query_rad"], df["qual"]):
            i += 1
            radius = float(rad)
            qual = int(qual)
            if qual != obj_qual:
                continue

            print("Querying {}...".format(objname))

            coordstring = "{} {}".format(ra, dec)
            c = get_coordinates(coordstring) # Safer

            tbl = Irsa.query_region(c, catalog=neowise, spatial="Cone",
                                radius=radius * u.arcsec)

            tbl = tbl.to_pandas()
            tbl = tbl.loc[tbl["qual_frame"]>= min_qual_frame]
            tbl = tbl[REQUIRED_COLS]
            
            
            cntr = (np.mean(tbl["ra"], axis=0), np.mean(tbl["dec"], axis=0))
            
            if kind == "null":
                key[cntr] = (0, objname)
            elif kind == "nova":
                key[cntr] = (1, objname)
            elif kind == "pulsating_var" or kind == "transit":
                key[cntr] = (2, objname)
            
            
            testdb = pd.concat((testdb, tbl), axis=0)

            os.system("clear")
            print("{}%".format(i*100 // len(df)), end="\r")
            print(kind + ": ", len(df.loc[df["qual"] >= min_qual_frame]))
            print("length ", len(tbl))

with open("cached_data/testdata.pkl", "wb") as f:
    pickle.dump(testdb, f)

with open("cached_data/key.pkl", "wb") as f:
    pickle.dump(key, f)
        
print("Number of examples:", len(key.keys()))
print("Total rows:", len(testdb))
print(testdb)

