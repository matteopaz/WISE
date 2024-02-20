
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
from faker import Faker
import os
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly_style import update_layout
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

ROOT = os.path.join("./")

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = nn.CrossEntropyLoss().to(device)

def save(model, kwargs, name):
  savestr = "state_dicts/model" + name
  with open(ROOT + savestr + ".pt", "wb") as f:
    torch.save(model.state_dict(),f)
  with open(ROOT + savestr + ".pkl", "wb") as f:
    pickle.dump(kwargs, f)

def gen_train(model, kwargs, optim, lossfn, trainloader, validloader, epochs, log=False, tracking=False):
  progress_bar = tqdm(total=epochs, desc="Training Progress")
  lastplottime = perf_counter()

  savename = Faker().name().replace(" ", "_")

  trainloss = []
  validloss = []

  predts = []
  nullts = []
  novats = []
  pulsatingts = []
  transitts = []
  accuracyts = []
  f1ts = []
  
  faileds = {}
  if tracking:
    for _, __, names in validloader:
      for name in names:
        faileds[name] = [0,None]
  
  for e in range(epochs):
    epoch_loss = []
    valid_loss = []

    epoch_preds = []

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

    model.train()
    for data, label in trainloader:
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

    model.eval()
    for example in validloader:
      
      if tracking:
        data, label, name = example
      else:
        data, label = example
      
      out = model(data)
      loss = loss_fn(out, label)
      valid_loss.append(loss.item())
      i = torch.argmax(out, dim=1).cpu()
      j = torch.argmax(label, dim=1).cpu()
      
      epoch_preds.append([i,j])

      if tracking:
        zipped = zip(name, i, j)
      else:
        zipped = zip([0] * len(i),i,j)

      for num, el in enumerate(zipped):
        idx = el[1]
        jdx = el[2]
        exs += 1
        if idx == jdx:
          correct += 1
        elif tracking:
          exname = el[0]
          faileds[exname][0] += 1
          if faileds[exname][1] == None:
            faileds[exname][1] = torch.zeros_like(out[num])
          
          faileds[exname][1][torch.argmax(out[num])] += 1

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

    y_true = torch.concat([j for i, j in epoch_preds], dim=0)
    y_pred = torch.concat([i for i, j in epoch_preds], dim=0)
    predts.append([y_true, y_pred])
    f1ts.append(f1_score(y_true, y_pred, average='macro'))
    conf = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(conf)

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

    if np.argmin(validloss) == len(validloss)-1 and e > 25:
      save(model, kwargs, savename)

    progress_bar.update(1)
    if perf_counter() - lastplottime > 1:
      p = getprogressplot(trainloss, validloss, f1ts, nullts, novats, pulsatingts, transitts, epochs, e)
      p = p.update_layout(title="Training Curve - {}".format(savename), xaxis_title="Epochs", yaxis_title="Loss")
      clear_output(wait=True)
      display(p)
      disp.plot()
      plt.show()
      lastplottime = perf_counter()
    
    # print("Most Failed: ", np.argmax(faileds))
    # print("Epoch ", e, ": ", training_loss_epoch)
    # print("nulls:", nullac, "novas: ", novacc, "pulsators: ", pulsatoracc, "transits: ", transitacc)

  x = range(epochs)

  p = getprogressplot(trainloss, validloss, accuracyts, nullts, novats, pulsatingts, transitts, epochs, e)
  p = p.update_layout(title="Training Curve - {}".format(savename, e, epochs), xaxis_title="Epochs", yaxis_title="Loss")
  if np.argmin(validloss) == len(validloss)-1 and e > 100:
    save(model, kwargs, savename)
  clear_output(wait=False)
  time.sleep(1)
  display(p)
  faileds = sorted(faileds.items(), key= lambda item: item[1][0], reverse=True)
  for name, val in faileds:
    if isinstance(val[1], torch.Tensor):
      val[1] = torch.trunc(val[1]).squeeze().tolist()
      # val[1] = nn.functional
    else:
      val[1] = 0
    
  print("faileds of valid: ", faileds)
  # print 3 most failed indexes in one line
  print("Epoch ", e, ": ", training_loss_epoch)

  save(model, kwargs, savename)

def itertrain(model, kwargs, optim, loss_fn, trainloader, validloader, epochs, itersperepoch=1, tracking=False):
  progress_bar = tqdm(total=epochs, desc="Training Progress")
  lastplottime = perf_counter()

  savename = Faker().name().replace(" ", "_")

  trainloss = []
  validloss = []

  predts = []
  nullts = []
  novats = []
  pulsatingts = []
  transitts = []
  accuracyts = []
  f1ts = []


  epoch_loss = []
  valid_loss = []

  epoch_preds = []

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

  faileds = {}
  if tracking:
    for _, __, names in validloader:
      for name in names:
        faileds[name] = [0,None]

  e = 0
  for data, label in trainloader:
    if torch.isnan(data).any():
      idkx = torch.where(torch.isnan(data))[0]
      # print(data[idkx])
      loc = torch.where(torch.isnan(data[idkx]))
      print("Data, Label")
      print(data[idkx].squeeze()[loc[1]-2:loc[1]+2])
      print(label[idkx].squeeze())
      raise ValueError("Nan in data")
    
    out = model(data)
    loss = loss_fn(out, label)
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    epoch_loss.append(loss.detach().item())

    if e % itersperepoch == 0:
      model.eval()
      valid_loss = []

      epoch_preds = []

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


      for example in validloader:
        data = example[0]
        label = example[1]
        if tracking:
          name = example[2]
        
        out = model(data)
        loss = loss_fn(out, label)
        valid_loss.append(loss.item())
        i = torch.argmax(out, dim=1).cpu()
        j = torch.argmax(label, dim=1).cpu()
        
        epoch_preds.append([i,j])

        if tracking:
          zipped = zip(name, i, j)
        else:
          zipped = zip([0] * len(i),i,j)

        for num, el in enumerate(zipped):
          idx = el[1]
          jdx = el[2]
          exs += 1
          if idx == jdx:
            correct += 1
          elif tracking:
            exname = el[0]
            faileds[exname][0] += 1
            if faileds[exname][1] == None:
              faileds[exname][1] = torch.zeros_like(out[num])
            
            faileds[exname][1][torch.argmax(out[num])] += 1

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

      y_true = torch.concat([j for i, j in epoch_preds], dim=0)
      y_pred = torch.concat([i for i, j in epoch_preds], dim=0)
      predts.append([y_true, y_pred])
      f1ts.append(f1_score(y_true, y_pred, average='macro'))
      conf = confusion_matrix(y_true, y_pred)
      disp = ConfusionMatrixDisplay(conf)

      training_loss_epoch = np.mean(epoch_loss)
      epoch_loss = []
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

      if np.argmin(validloss) == len(validloss)-1 and e > 25*itersperepoch:
        save(model, kwargs, savename)

      progress_bar.update(1)
      if perf_counter() - lastplottime > 1:
        p = getprogressplot(trainloss, validloss, f1ts, nullts, novats, pulsatingts, transitts, epochs, e // itersperepoch)
        p = p.update_layout(title="Training Curve - {}".format(savename), xaxis_title="Epochs", yaxis_title="Loss")
        clear_output(wait=True)
        display(p)
        disp.plot()
        plt.show()
        lastplottime = perf_counter()

      
      print("top 5 faileds of valid: ", sorted(faileds.items(), key= lambda item: item[1][0], reverse=True)[:5])
      
      torch.cuda.empty_cache()
      model.train()
    e += 1
  
  # print("Most Failed: ", np.argmax(faileds))
  # print("Epoch ", e, ": ", training_loss_epoch)
  # print("nulls:", nullac, "novas: ", novacc, "pulsators: ", pulsatoracc, "transits: ", transitacc)

  faileds = sorted(faileds.items(), key= lambda item: item[1][0], reverse=True)
  for name, val in faileds:
    if isinstance(val[1], torch.Tensor):
      val[1] = torch.trunc(val[1]).squeeze().tolist()
      # val[1] = nn.functional
    else:
      val[1] = 0
    
  print("faileds of valid: ", faileds)

  p = getprogressplot(trainloss, validloss, accuracyts, nullts, novats, pulsatingts, transitts, epochs, epochs)
  p = p.update_layout(title="Training Curve - {}".format(savename, epochs, epochs), xaxis_title="Epochs", yaxis_title="Loss")
  clear_output(wait=False)
  display(p)

  save(model, kwargs, savename)


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


      for example in validloader:
        data = example[0]
        label = example[1]
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
      