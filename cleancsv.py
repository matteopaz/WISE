import pickle
import pandas as pd

file = "./runner/inp/1deg"

dataframe = pd.read_csv(file + ".csv")

dataframe = dataframe[dataframe["qual_frame"] >= 9]
dataframe = dataframe[["ra", "dec", "w1mpro", "mjd"]].copy()
dataframe.dropna(axis=0, how="any", inplace=True)

# overwrite the dataframe with the new one as CSV

dataframe.to_csv(file + "_clean.csv", index=True)