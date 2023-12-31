
import torch
import torch.nn as nn
import tqdm
from tqdm.auto import tqdm
import numpy as np
from helpers import getprogressplot
from time import perf_counter
from IPython.display import display, clear_output
from datetime import datetime
import pickle
import os
import time
import plotly.graph_objects as go
from plotly_style import update_layout

ROOT = os.path.join("./")

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = nn.CrossEntropyLoss().to(device)

def gen_train(model, kwargs, optim, lossfn, trainloader, validloader, epochs, log=False):
  progress_bar = tqdm(total=epochs, desc="Training Progress")

  trainloss = []
  validloss = []

  nullts = []
  novats = []
  pulsatingts = []
  transitts = []
  accuracyts = []
  
  num_of_valids = np.sum([s[0].shape[0] for s in validloader]) # works only if batched
  faileds = [0 for _ in range(num_of_valids)]

  for e in range(epochs):
    epoch_loss = []
    valid_loss = []

    null_correct = 0
    nova_correct = 0
    pulsating_correct = 0
    transit_correct = 0
    correct = 0


    novas = 0
    pulsators = 0
    transits = 0
    nulls = 0
    exs = 0
    
    for data, label in trainloader:
      model.train()
      if log:
        print("Data", data)
        print("Label", label)
      out = model(data)
      if log:
        print("Out", out)
      loss = loss_fn(out, label)    
      if log:
        print(loss)

      epoch_loss.append(loss.item())

      loss.backward()
      optim.step()
      optim.zero_grad()


    for data, label in validloader:
      model.eval()
      out = model(data)
      loss = loss_fn(out, label)
      valid_loss.append(loss.item())
      i = torch.argmax(out, dim=1).cpu()
      j = torch.argmax(label, dim=1).cpu()

      for ex, (idx, jdx) in enumerate(zip(i,j)):
        exs += 1
        if idx == jdx:
          correct += 1
        else:
          faileds[ex] += 1

        if jdx == 0:
          nulls += 1
          if idx == jdx:
            null_correct += 1

        if jdx == 1:
          novas += 1
          if idx == jdx:
            nova_correct +=1
        if jdx == 2:
          pulsators += 1
          if idx == jdx:
            pulsating_correct +=1
        if jdx == 3:
          transits += 1
          if idx == jdx:
            transit_correct +=1

    training_loss_epoch = np.mean(epoch_loss)
    validation_loss_epoch = np.mean(valid_loss)

    nullac = null_correct / (nulls + 0.000001)
    novacc = nova_correct / (novas + 0.000001)
    pulsatoracc = pulsating_correct / (pulsators + 0.00001)
    transitacc = transit_correct / (transits + 0.00001)
    accuracy = correct / exs

    trainloss.append(training_loss_epoch)
    validloss.append(validation_loss_epoch)

    nullts.append(nullac)
    novats.append(novacc)
    pulsatingts.append(pulsatoracc)
    transitts.append(transitacc)
    accuracyts.append(accuracy)


    progress_bar.update(1)
    p = getprogressplot(trainloss, validloss, accuracyts, nullts, novats, pulsatingts, transitts, epochs, e)
    clear_output(wait=False)
    display(p)
    print("Most Failed: ", np.argmax(faileds))

    print("Epoch ", e, ": ", training_loss_epoch)
    print("nulls:", nullac, "novas: ", novacc, "pulsators: ", pulsatoracc, "transits: ", transitacc)

  x = range(epochs)

  p = getprogressplot(trainloss, validloss, accuracyts, nullts, novats, pulsatingts, transitts, epochs, e)
  clear_output(wait=True)
  time.sleep(1)
  display(p)
  print("faileds of valid: ", faileds)
  # print 3 most failed indexes in one line
  
  print("Most Failed: ", np.argsort(faileds)[-3:])
  print("Epoch ", e, ": ", training_loss_epoch)

  savestr = "state_dicts/model" + datetime.now().strftime("%Y%m%d-%H%M%S")
  with open(ROOT + savestr + ".pt", "wb") as f:
    torch.save(model.state_dict(),f)

  with open(ROOT + savestr + ".pkl", "wb") as f:
    pickle.dump(kwargs, f)


  print("saved as {}".format(savestr))


def compete(trainers, epochs, trainloader, validloader, log=False):
  # trainers are model-optimizer pairs

  progress_bar = tqdm(total=epochs, desc="Training Progress")
  loss_fn = nn.CrossEntropyLoss().to(device)


  trainloss = {}
  validloss = {}
  accuracy = {}
  
  for name, _model, _optim in trainers:
    trainloss[name] = []
    validloss[name] = []
    accuracy[name] = []


  for e in range(epochs):
    t1 = perf_counter()

    for name, model, optim in trainers:
      epoch_loss = []
      valid_loss = []

      correct = 0
      exs = 0

      for data, label in trainloader:
        model.train()
        out = model(data)

        optim.zero_grad()

        loss = loss_fn(out, label)
        epoch_loss.append(loss.item())

        loss.backward()
        if log:
          print(out, label)
          print(loss)
        optim.step()


      for data, label in validloader:
        model.eval()
        out = model(data)
        loss = loss_fn(out, label)
        valid_loss.append(loss.item())
        i = torch.argmax(out, dim=1).cpu()
        j = torch.argmax(label, dim=1).cpu()
        for idx, jdx in zip(i,j):
          exs += 1
          if idx == jdx:
            correct += 1

      training_loss_epoch = np.mean(epoch_loss)
      validation_loss_epoch = np.mean(valid_loss)

      accuracy_epoch = correct / exs

      trainloss[name].append(training_loss_epoch)
      validloss[name].append(validation_loss_epoch)
      accuracy[name].append(accuracy_epoch)
      
    traces = []
    colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, (name, _, __) in enumerate(trainers):
      color = colors[3*i % len(colors): 3*(i+1) % len(colors)]
      tr1 = go.Scatter(x=list(range(e)), y=trainloss[name], mode='lines', name=f'Training Loss --{name}', line=dict(color=color[0]), opacity=0.8)
      tr2 = go.Scatter(x=list(range(e)), y=validloss[name], mode='lines', name=f'Validation Loss --{name}', line=dict(color=color[1]), opacity=0.75)
      tr3 = go.Scatter(x=list(range(e)), y=accuracy[name], mode='lines', name=f'Validation Accuracy --{name}', line=dict(color=color[2]), opacity=0.5)
      traces += [tr1, tr2, tr3]
      
    fig = go.Figure(data=traces)
    fig.update_layout(title=f"Epoch {e}", xaxis_title="Epochs", yaxis_title="Loss", width=1000)
    clear_output(wait=True)
    fig.show()
      