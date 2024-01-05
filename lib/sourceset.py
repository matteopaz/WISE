from copy import copy
import random
import torch
from torch.utils.data import Dataset
# import blank dict
from collections import defaultdict


def swap(pct):
  def inner(data):
    new = torch.clone(data)
    n = int(pct*len(new)) // 2
    for _ in range(n):
      one, two = random.sample(range(len(new)), 2)

      temp = torch.clone(new[one])

      new[one] = new[two]
      new[two] = temp
    return new
  return inner


def flip_x(data):
  return torch.flip(data, (0,))

def flip_y(data):
  new = torch.clone(data)
  new[:, 0] = -1 * new[:, 0]
  new[:, 2] = -1 * new[:, 2]
  return new

def resample(std, pct=0.25):
  def inner(data):
    new = torch.clone(data)
    stddev = [0 for _ in range(len(new))]

    for i in random.sample(range(len(stddev)), int(pct*len(stddev))):
      stddev[i] = std

    stddev = torch.tensor(stddev)

    new[:, 0] = torch.normal(new[:, 0], stddev)
    new[:, 2] = torch.normal(new[:, 2], stddev)
    return new

  return inner

def rescale_x(data):
  s = random.random() *1.5

  new = torch.clone(data)

  new[:, -1] = s * new[:, -1]

  return new

def rescale_y(data):
  s = random.random() * 1.5

  new = torch.clone(data)

  new[:, 0] = s * new[:, 0]
  new[:, 2] = s * new[:, 2]

  return new

class SourceSet(Dataset):
  def __init__(self, buckets, augmentation=None, choose=3, augmentation_frac=3, equalize=False, point_limit=1000):

    self.choose = choose
    self.augmentation_frac = augmentation_frac

    self.buckets = defaultdict(list)
    for kind in buckets:
        for obj in buckets[kind]:
            if len(obj) < point_limit:
                self.buckets[kind].append(obj.to_tensor())
            else:
                newobj = obj.get_subset(len(obj) - point_limit, len(obj))
                self.buckets[kind].append(newobj.to_tensor())
                

    self.all = []

    if augmentation:
      for kind in self.buckets:
        new = []
        new += self.buckets[kind]
        for ex in self.buckets[kind]:
          for _ in range(self.augmentation_frac):
            new += self.apply_pipeline([ex], random.sample(augmentation, choose))

        self.buckets[kind] = new

      # Equalization
      if equalize:
        lens = [len(self.buckets[b]) for b in self.buckets]
        makeup = [max(lens) - v for v in lens]

        for i, count in enumerate(makeup):
          key = list(self.buckets.keys())[i]
          for ex in self.buckets[key]:
            if count <= 0:
              break
            self.buckets[key] += self.apply_pipeline([ex], random.sample(augmentation, choose))
            count -= 1


    for kind in self.buckets: # Data, Label pairing
      for i, ex in enumerate(self.buckets[kind]):

        label = torch.zeros(len(self.buckets.keys()))
        label[list(self.buckets.keys()).index(kind)] = 1

        self.all.append((ex, label))
        self.buckets[kind][i] = ((ex, label))

  def apply_pipeline(self, examples, pipeline):

    p = copy(pipeline)
    if len(p) == 0:
      return examples
    new = []
    fn, times = p.pop(0)
    for ex in examples:
      for _ in range(times):
        new.append(fn(ex))
      del ex

    return self.apply_pipeline(new, p)

  def get_buckets(self):
    return self.buckets

  def __getitem__(self, idx):
    # Commented out is random sampling
    # key = random.choice(list(self.buckets.keys()))
    # item = random.choice(self.buckets[key])
    # return item
    return self.all[idx]

  def __len__(self):
    return len(self.all)


default_aug = [(rescale_x, 1), (rescale_y, 1), (resample(0.65, 0.35), 1), (flip_x, 1), (swap(0.1), 1), (flip_y, 1), (swap(0.2), 1), (resample(1, 0.25), 1)]
