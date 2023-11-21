
import torch
import torch.nn as nn
import tqdm
from tqdm.auto import tqdm
import numpy as np
from helpers import getprogressplot
from time import perf_counter
from IPython.display import display, clear_output

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = nn.CrossEntropyLoss().to(device)

def compete(trainers, epochs, trainloader, validloader, log=False):
  # trainers are model-optimizer pairs

  progress_bar = tqdm(total=epochs, desc="Training Progress")
  loss_fn = nn.CrossEntropyLoss().to(device)


  trainloss = {}
  validloss = {}

  nullts = {}
  novats = {}
  pulsatingts = {}
  transitts = {}
  accuracyts = {}

  for name, _model, _optim in trainers:
    trainloss[name] = []
    validloss[name] = []
    nullts[name] = []
    novats[name] = []
    pulsatingts[name] = []
    transitts[name] = []
    accuracyts[name] = []


  for e in range(epochs):
    t1 = perf_counter()


    for name, model, optim in trainers:
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
          exs+=1
          if idx == jdx:
            correct += 1

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

      trainloss[name].append(training_loss_epoch)
      validloss[name].append(validation_loss_epoch)

      nullts[name].append(nullac)
      novats[name].append(novacc)
      pulsatingts[name].append(pulsatoracc)
      transitts[name].append(transitacc)
      accuracyts[name].append(accuracy)



    dt = perf_counter() - t1
    if dt < 0.65:
      if e % int(1.75/dt) == 0:
        for name, _, __ in trainers:
          clear_output()
          fig = getprogressplot(trainloss[name], validloss[name], accuracyts[name], nullts[name], novats[name], pulsatingts[name], transitts[name], epochs, e)
          fig.update_layout(title=name+' Statistics: {}/{}'.format(e, epochs),
                        xaxis_title='Epochs',
                        yaxis_title='Loss',
                        width=650)
          display(fig)
    else:
      for name, _, __ in trainers:
        clear_output()
        fig = getprogressplot(trainloss[name], validloss[name], accuracyts[name], nullts[name], novats[name], pulsatingts[name], transitts[name], epochs, e)
        fig.update_layout(title=name+' Statistics: {}/{}'.format(e, epochs),
                      xaxis_title='Epochs',
                      yaxis_title='Loss',
                      width=1000)
        display(fig)

    progress_bar.update(1)