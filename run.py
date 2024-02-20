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

torch.set_default_device("cuda")

testing = False 
name = "nancy"
inp = "1deg"

# For Testing
print("Loading data...")
t1 = perf_counter()

data_in = pd.read_csv("./runner/inp/{}.csv".format(inp))

with open("./runner/cluster/{}_ptrs.pkl".format(inp), "rb") as f:
    ptrs = pickle.load(f)
    


print(f"Elapsed -- {perf_counter()-t1}")
    
batches = BatchGen(data_in, ptrs, batchsize=10)
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
    # maxes = 0.5 * (torch.pow(maxes, 160) + torch.pow(maxes, 5)) # rescaling confidence
    classified = torch.argmax(out, dim=1).cpu().numpy()
    
    for i, center in enumerate(centers):
        founds[classified[i]].append((maxes[i], center))

for kind in founds:
    founds[kind] = sorted(founds[kind], key=lambda item: item[0], reverse=True)

print(f"Elapsed -- {perf_counter()-t1}")
print("Outputting...")

with open("./runner/null.txt", "w") as f:
    for conf, (ra, dec) in founds[0]:
        f.write("{} {} --- {}\n".format(ra, dec, conf))

with open("./runner/nova.txt", "w") as f:
    for conf, (ra, dec) in founds[1]:
        f.write("{} {} --- {}\n".format(ra, dec, conf))
        
with open("./runner/pulsating_var.txt", "w") as f:
    for conf, (ra, dec) in founds[2]:
        f.write("{} {} --- {}\n".format(ra, dec, conf))

with open("./runner/transit.pkl", "wb") as f:
    for conf, (ra, dec) in founds[3]:
        f.write("{} {} --- {}\n".format(ra, dec, conf))
