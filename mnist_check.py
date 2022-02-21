import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import pdb
from model import *
import argparse


#task = "CLearn"
task = "CLearn_Ben"

## load mnist dataset
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

root = './mnist_data'
if not os.path.exists(root):
    os.mkdir(root)




trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = MNIST(root=root, train=True, transform=trans, download=True)
test_set = MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 128
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

## network

class Net(nn.Module):
    def __init__(self, task):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.fc3 = nn.Linear(20, 2)
        self.task = task

    def forward(self, state, desc):
        desc = torch.nn.functional.one_hot(desc)
        x = F.relu(self.conv1(state))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if self.task == "CLearn":
            x = torch.cat([x, desc], dim = 1)
            x = self.fc3(x)
            #x = torch.sigmoid(x)
        return x


class BenModel(nn.Module):
    def __init__(self):
        super(BenModel, self).__init__()
        self.fc1 = nn.Linear(794,128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        output = torch.relu(self.fc1(x))
        output = self.fc2(output)
        output = torch.softmax(output, dim = 1)
        return output






model = BenModel()

if use_cuda:
    model = model.cuda()

if "CLearn" in task:
    ## training
    learning_rate = 0.0003
    epochs =  50
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    #loss_fn = torch.nn.BCELoss()
    loss_fn = nn.CrossEntropyLoss()

if task == "Classification":
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    

for epoch in range(10):
    losses = []
    accur = []
    prop = []
  
    # trainning
    ave_loss = 0
    for batch_idx, (state, desc) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            state, desc = state.cuda(), desc.cuda()
        og_state, desc = Variable(state), Variable(desc)


        if task == "CLearn":
            state_incorrect = state.clone()
            desc_incorrect = torch.randint(1, 10, desc.shape).cuda()
            
            label_correct = torch.ones_like(desc).cuda()
            label_incorrect = torch.zeros_like(desc).cuda()

            label = torch.cat([label_correct,label_incorrect]).float()
            desc = torch.cat([desc,desc_incorrect])
            state = torch.cat([state,state_incorrect])

            # #shuffle
            # shuffle_idx = torch.randperm(state.shape[0])
            # state = state[shuffle_idx]
            # desc = desc[shuffle_idx]
            # label = label[shuffle_idx]
            
            output = model(state, desc)

            loss = loss_fn(output.squeeze(), label.long())

            
            
            #predicted = (output>0.5).float()
            #loss = loss_fn(predicted.squeeze(), label)
            #print(predicted)
            # prop_ones =  predicted.mean().detach()
            predictions = torch.argmax(output, dim = 1) 
            acc = torch.mean((predictions == label).float())
            #acc = torch.mean((predicted.squeeze() == label).float())
            #print(loss)

        if task == "CLearn_Ben":
            state = torch.reshape(og_state, (-1,784))
            state_incorrect = state.clone()
            desc_incorrect = torch.randint(1, 10, desc.shape).cuda()
            
            label_correct = torch.ones_like(desc).cuda()
            label_incorrect = torch.zeros_like(desc).cuda()

            label = torch.cat([label_correct,label_incorrect]).float()
            desc = torch.cat([desc,desc_incorrect])
            state = torch.cat([state,state_incorrect])
            desc = torch.nn.functional.one_hot(desc)

            try:
                x = torch.cat([state,desc], dim = 1)
            except Exception:
                pdb.post_mortem()
            pdb.set_trace()
            output = model(x)
            loss = loss_fn(output.squeeze(), label.long())

            
            
            #predicted = (output>0.5).float()
            #loss = loss_fn(predicted.squeeze(), label)
            #print(predicted)
            # prop_ones =  predicted.mean().detach()
            predictions = torch.argmax(output, dim = 1) 
            acc = torch.mean((predictions == label).float())
            #acc = torch.mean((predicted.squeeze() == label).float())
            #print(loss)


        if task == "Classification":
            output = model(state, desc)
            pdb.set_trace
            loss = loss_fn(output.squeeze(), desc)
            acc = torch.mean((torch.argmax(output, dim = 1) == desc).float())
            
        
         # TODO:  WRITE MODEL
        #calculate loss
        

        #accuracy
        

        #backprop
        
        loss.backward()
        optimizer.step()
        #prop.append(prop_ones)
        losses.append(loss.detach())
        accur.append(acc.detach())
    
    epoch_loss = torch.mean(torch.stack(losses))
    epoch_acc = torch.mean(torch.stack(accur))
    #epoch_prop = torch.mean(torch.stack(prop))
    print("epoch:", epoch)
    #print("prop:", epoch_prop)
    print("accuracy:", epoch_acc)
    print("loss:", epoch_loss)

    # # testing
    # correct_cnt, ave_loss = 0, 0
    # total_cnt = 0
    # for batch_idx, (x, target) in enumerate(test_loader):
    #     if use_cuda:
    #         x, target = x.cuda(), target.cuda()
    #     x, target = Variable(x, volatile=True), Variable(target, volatile=True)
    #     out = model(x)
    #     loss = criterion(out, target)
    #     _, pred_label = torch.max(out.data, 1)
    #     total_cnt += x.data.size()[0]
    #     correct_cnt += (pred_label == target.data).sum()
    #     # smooth average
    #     ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        
    #     if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
    #         print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
    #             epoch, batch_idx+1, ave_loss, correct_cnt * 1.0 / total_cnt))

# torch.save(model.state_dict(), model.name())



