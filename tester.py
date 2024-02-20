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

name = "modelSean_Simpson"

# For Testing
print("Loading data...")
t1 = perf_counter()
with open("./cached_data/testdata.pkl", "rb") as f:
    data_in = pickle.load(f)

data_in.dropna(axis=0, how="any", inplace=True)

print(f"Elapsed -- {perf_counter()-t1}")

print("Clustering...")
t1 = perf_counter()
labelmap = get_labels(data_in)
print(f"Elapsed -- {perf_counter()-t1}")
print("Making Batches...")
t1 = perf_counter()
batches = BatchGen(data_in, labelmap, batchsize=1)
print(f"Elapsed -- {perf_counter()-t1}")
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

maxdist = 3/3600 # arcsecs from correct center
with open("./cached_data/key.pkl", "rb") as f:
    key = pickle.load(f)
    
total = len(key)
correct = 0

pred_y = []
true_y = []
for classif in founds:
    for confidence, foundcenter in founds[classif]:
        for center, (kind, name) in key.items():
            if ((foundcenter[0] - center[0])**2 + (foundcenter[1] - center[1])**2) < maxdist**2:
                pred_y.append(classif)
                true_y.append(kind)
                if classif == kind:
                    print(f"Got {name} correct as {kind}")
                    correct += 1
                else:
                    print(f"Got {name} incorrectly classified as {classif}, was actually {kind} - conf: {int(confidence*1000) / 1000}")
print(f"Total Accuracy: {correct * 100 / total}%")

conf = confusion_matrix(true_y, pred_y)
disp = ConfusionMatrixDisplay(conf)
disp.plot()
plt.savefig("./temp.png")
