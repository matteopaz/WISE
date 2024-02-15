import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np

class PseudoSet(Dataset):
    def __init__(self, buckets, valid=False):
        self.all = []
        self.lengrouped = defaultdict(list)
        self.valid = valid

        for kind in buckets:
            for ex in buckets[kind]:
                x = ex[0]
                y = ex[1]
                # norm and stuff
                l = len(x)

                snr = -torch.log10(torch.tensor(y)) / 3
                label = torch.zeros(len(buckets.keys()))
                label[list(buckets.keys()).index(kind)] = 1
                y = torch.tensor((y - np.mean(y)) / np.std(y))
                y = torch.arcsinh(y) # compress with arcsinh

            
                x = torch.tensor((x - np.min(x)) / np.max(x))
                tens = torch.stack([y, snr, x], dim=0).T.to(torch.float32) # precision not needed after this normalization

                if not valid:
                    self.all.append((tens, label))
                    self.lengrouped[l].append((tens, label))
                else:
                    self.all.append((tens, label, "unnamed"))
                    self.lengrouped[l].append((tens, label, "unnamed"))



    def __len__(self):
        return len(self.lengrouped.keys())
        # return len(self.all)
    
    def __getitem__(self, idx):
        oflen = self.lengrouped[list(self.lengrouped.keys())[idx]]
        batch = torch.stack([x[0] for x in oflen], dim=0)
        labels = torch.stack([x[1] for x in oflen], dim=0)
        if self.valid:
            names = [x[2] for x in oflen]
            return batch, labels, names
        else:
            return batch, labels
        # return self.all[idx]