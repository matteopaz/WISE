import pickle
import pandas as pd
import torch
import numpy as np
from runner_backend import BatchGen, get_labels
from time import perf_counter
from torch.nn.functional import softmax
from models.fourier import CNFourierModel
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import json

torch.set_default_device("cuda")

name = "GOOD"
inp = "orionsbelt25deg"

# For Testing
print("Loading data...")
t1 = perf_counter()

data_in = pd.read_csv("./runner/inp/{}_clean.csv".format(inp))

with open("./runner/cluster/{}_ptrs.pkl".format(inp), "rb") as f:
    ptrs = pickle.load(f)

# print lengths of clusters in descending size
# lens = sorted([len(v) for v in ptrs.values()], reverse=True)
# print(lens)


# ptrs = json.load(open("./runner/cluster/{}_ptrs.json".format(inp), "r"))

print(f"Elapsed -- {perf_counter()-t1}")
    
batches = BatchGen(data_in, ptrs, batchsize=128)
print("Readying Model...")

with open(f"./state_dicts/{name}.pkl", "rb") as f:
    params = pickle.load(f)

model = CNFourierModel(**params).cuda()
with open(f"./state_dicts/{name}.pt", "rb") as f:
    model.load_state_dict(torch.load(f))
model.eval().cuda()

founds = {0:[], 1:[], 2:[], 3:[]}
print("Evaluating with Model..")
t1 = perf_counter()
for centers, batch in batches:
    # NANS IN DATASET
    out = softmax(model(batch), dim=1).detach()
    maxes = torch.max(out, dim=1).values.cpu().numpy()
    classified = torch.argmax(out, dim=1).cpu().numpy()
    confused = np.where(maxes < 0.8)
    
    for i, center in enumerate(centers):
        if i in confused[0]:
            founds[3].append((maxes[i], center))
        else:
            founds[classified[i]].append((maxes[i], center))


founds[1] = sorted(founds[1], key=lambda item: item[0], reverse=True)
founds[2] = sorted(founds[2], key=lambda item: item[0], reverse=True)
founds[3] = sorted(founds[3], key=lambda item: item[0])


print(f"Elapsed -- {perf_counter()-t1}")
print("Outputting...")

with open("./runner/null.txt", "w") as f:
    f.write("Found {} nulls".format(len(founds[0])))

with open("./runner/nova.txt", "w") as f:
    for conf, (ra, dec) in founds[1]:
        f.write("{} {} --- {}\n".format(ra, dec, conf))
        
with open("./runner/var.txt", "w") as f:
    for conf, (ra, dec) in founds[2]:
        f.write("{} {} --- {}\n".format(ra, dec, conf))

with open("./runner/confused.txt", "w") as f:
    for conf, (ra, dec) in founds[3]:
        f.write("{} {} --- {}\n".format(ra, dec, conf))
