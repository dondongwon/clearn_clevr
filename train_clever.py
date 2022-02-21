import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np

import pdb
import statistics
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import json

from model import *


state_input = 'Image'
use_action = True

#description 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

class dataset(Dataset):
  def __init__(self, buffer_state, buffer_desc, geometric_sampling = True):

    self.state = buffer_state
    self.desc = buffer_desc

    buffer_size = buffer_state.shape[0]

    self.traj_len = self.state.shape[1]
    self.length = self.state.shape[0]
    #self.geometric_sampling = geometric_sampling
    self.gamma = 0.9

  def geometric_sampling_fn(self):
    start_index = np.random.randint(0, self.traj_len)
    pos_index = np.random.geometric(1 - self.gamma) + start_index

    if pos_index >= 100:
      return self.geometric_sampling_fn()
    else:
      return start_index, pos_index

  def __getitem__(self,idx):
    
    start_index, pos_index = self.geometric_sampling_fn()

    state = self.state[idx, start_index, ...]
    desc = self.desc[idx, pos_index, ...]

    return state, desc

  def __len__(self):
    return self.length


class ActionDataset(Dataset):
  def __init__(self, buffer_state, buffer_desc, buffer_action, geometric_sampling = True):

    self.state = buffer_state
    self.desc = buffer_desc
    self.action = buffer_action

    buffer_size = buffer_state.shape[0]

    self.traj_len = self.state.shape[1]
    self.length = self.state.shape[0]
    #self.geometric_sampling = geometric_sampling
    self.gamma = 0.5

  def geometric_sampling_fn(self):
    start_index = np.random.randint(0, self.traj_len)
    pos_index = np.random.geometric(1 - self.gamma) + start_index

    if pos_index >= 100:
      return self.geometric_sampling_fn()
    else:
      return start_index, pos_index

  def __getitem__(self,idx):
    
    start_index, pos_index = self.geometric_sampling_fn()


    state = self.state[idx, start_index, ...]
    action = self.action[idx, start_index, ...]
    desc = self.desc[idx, pos_index, ...]

    return state, desc, action

  def __len__(self):
    return self.length







# Model , Optimizer, Loss

if state_input == "State":
  model = ClassifierStateBB()
  train_state = torch.load('state.pt')[:9000]
  train_desc = torch.load('desc.pt')[:9000]
  pdb.set_trace()
  val_state = torch.load('state.pt')[9000:]
  val_desc = torch.load('desc.pt')[9000:]
if state_input == 'Image':
  
  state_ds = np.load('CRstateImage1k.npy')
  desc_ds = np.load('CRdescImage1k.npy')
  action_ds = np.load('CRactionImage1k.npy')
  ds_split = 900
  train_action = action_ds[:ds_split]
  train_state = state_ds[:ds_split]
  train_desc = desc_ds[:ds_split]
  val_action = action_ds[ds_split:]
  val_state = state_ds[ds_split:]
  val_desc = desc_ds[ds_split:]

  if use_action:
    model = ClassifierCRActionResnet()
    trainset = ActionDataset(train_state, train_desc, train_action)
    valset = ActionDataset(val_state, val_desc, train_action)
    valloader = DataLoader(valset, batch_size=32,shuffle=True)

  else:
    model = ClassifierCR()
    trainset = dataset(train_state, train_desc)
    valset = dataset(val_state, val_desc)
    valloader = DataLoader(valset, batch_size=32,shuffle=True)

if use_cuda:
  model = model.cuda()




learning_rate = 0.0001
epochs =  10000
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
pos_weight = torch.ones([1]).to(device)
loss_fn = nn.CrossEntropyLoss()
batch_size = 32

for epoch in range(epochs):
  losses = []
  accur = []
  prop = []
  
  trainloader = DataLoader(trainset, batch_size=32,shuffle=True)
 
  for j,batch in enumerate(tqdm(trainloader)):

    if use_action:
      state, desc, action = batch
      state = state.to(device)
      desc = desc.to(device)
      action = action.to(device)
      label = torch.ones(state.shape[0]*2).to(device)
      label[state.shape[0]:] = 0

      idx = torch.randperm(desc.shape[0])
      incorrect_desc = desc[idx].view(desc.size())

      state = torch.cat((state,state), dim = 0).to(device)
      action = torch.cat((action,action), dim = 0).to(device)
      desc = torch.cat((desc,incorrect_desc), dim = 0).to(device)
    else: 
      state, desc = batch
      state = state.to(device)
      desc = desc.to(device)
    
      label = torch.ones(state.shape[0]*2).to(device)
      label[state.shape[0]:] = 0

      idx = torch.randperm(desc.shape[0])
      incorrect_desc = desc[idx].view(desc.size())
    
      state = torch.cat((state,state), dim = 0).to(device)
      desc = torch.cat((desc,incorrect_desc), dim = 0).to(device)
    
    #desc = desc.sum(dim=1)
    #desc = torch.nn.functional.one_hot(desc, num_classes=len(desc_key))

    if use_action:
      output = model(state, desc, action)
    else: 
      output = model(state, desc)
    #calculate loss
    loss = loss_fn(output.squeeze(), label.long())
    #accuracy
    predictions = torch.argmax(output, dim = 1) 
    acc = torch.mean((predictions == label).float())
    prop_ones =  predictions.float().mean().detach()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    prop.append(prop_ones)
    losses.append(loss.detach())
    accur.append(acc.detach())

  epoch_loss = torch.mean(torch.stack(losses))
  epoch_acc = torch.mean(torch.stack(accur))
  epoch_prop = torch.mean(torch.stack(prop))
  
  

  with torch.set_grad_enabled(False):
    val_losses = []
    val_accur = []
    val_prop = []
    for i,batch in enumerate(tqdm(valloader)):
      if use_action:
        state, desc, action = batch
        state = state.to(device)
        desc = desc.to(device)
        action = action.to(device)
        
        label = torch.ones(state.shape[0]*2).to(device)
        label[state.shape[0]:] = 0

        idx = torch.randperm(desc.shape[0])
        incorrect_desc = desc[idx].view(desc.size())
        state = torch.cat((state,state), dim = 0).to(device)
        action = torch.cat((action,action), dim = 0).to(device)
        desc = torch.cat((desc,incorrect_desc), dim = 0).to(device)
      else: 
        state, desc = batch
        state = state.to(device)
        desc = desc.to(device)
        
        label = torch.ones(state.shape[0]*2).to(device)
        label[state.shape[0]:] = 0

        idx = torch.randperm(desc.shape[0])
        incorrect_desc = desc[idx].view(desc.size())
        
        state = torch.cat((state,state), dim = 0).to(device)
        desc = torch.cat((desc,incorrect_desc), dim = 0).to(device)
    
      
      #desc = desc.sum(dim=1)
      #desc = torch.nn.functional.one_hot(desc, num_classes=len(desc_key))

      if use_action:
        output = model(state, desc, action)
      else: 
        output = model(state, desc)
      #calculate loss

      val_loss = loss_fn(output.squeeze(), label.long())
      #accuracy
      val_predicted = torch.argmax(output, dim = 1) 
      val_acc = torch.mean((val_predicted == label).float())
      val_prop_ones =  val_predicted.float().mean().detach()

      val_acc = torch.mean((val_predicted.squeeze() == label).float())
      val_losses.append(val_loss.detach())
      val_prop.append(val_prop_ones)
      val_accur.append(val_acc.detach())
    
    #pdb.set_trace()
    epoch_val_loss = torch.mean(torch.stack(val_losses))
    epoch_val_acc = torch.mean(torch.stack(val_accur))
    epoch_val_prop = torch.mean(torch.stack(val_prop))
#   print("epoch:", epoch)
#   print("prop:", epoch_prop)
#   print("accuracy:", epoch_acc)
#   print("loss:", epoch_loss)
#   print("val_prop:", epoch_val_prop)
#   print("val_accuracy:", epoch_val_acc)
#   print("val_loss:", epoch_val_loss)
  writer.add_scalar("Accuracy/train", epoch_acc, epoch)
  writer.add_scalar("Loss/train", epoch_loss, epoch)
  writer.add_scalar("Accuracy/test", epoch_val_acc, epoch)
  writer.add_scalar("Loss/test", epoch_val_loss, epoch)


torch.save(model.state_dict(), "./model_weights/{}_smallaction_fix.pth".format(model.__class__.__name__))
