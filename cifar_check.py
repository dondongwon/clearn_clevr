import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3072+10,1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, state, desc):
        state = torch.reshape(state, (-1,3072))
        desc_new = torch.nn.functional.one_hot(desc, 10)
        x = torch.cat([state, desc_new], dim = 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.softmax(x, dim = 1)
        return x


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
model = SimpleModel().cuda()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
epochs =  40
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    losses = []
    accur = []
    prop = []
    running_loss = 0.0
    for i, data in (enumerate(trainloader, 0)):
        
        state, desc = data

        state_incorrect = state.clone()
        desc_incorrect = torch.randint(1, 10, (desc.shape[0],))
        label_correct = torch.ones_like(desc)
        label_incorrect = torch.zeros_like(desc)

        label = torch.cat([label_correct,label_incorrect])
        desc = torch.cat([desc,desc_incorrect])
        state = torch.cat([state,state_incorrect])

        label = label.cuda()
        state  = state.cuda()
        desc = desc.cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(state, desc)

        



        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # save statistics
        running_loss += loss.item()
        if i%200 == 199:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
        predictions = torch.argmax(output, dim = 1) 
        acc = torch.mean((predictions == label).float())
        prop_ones =  predictions.float().mean().detach()

        prop.append(prop_ones)
        losses.append(loss.detach())
        accur.append(acc.detach())

    with torch.set_grad_enabled(False):
        val_losses = []
        val_accur = []
        val_prop = []
        for i,batch in enumerate(testloader, 0):
            state, desc = batch

            state_incorrect = state.clone()
            desc_incorrect = torch.randint(1, 10, desc.shape)
            label_correct = torch.ones_like(desc)
            label_incorrect = torch.zeros_like(desc)

            label = torch.cat([label_correct,label_incorrect])
            desc = torch.cat([desc,desc_incorrect])
            state = torch.cat([state,state_incorrect])

            label = label.cuda()
            state  = state.cuda()
            desc = desc.cuda()
        
            output = model(state, desc)


            #calculate loss
            try: 
                val_loss = criterion(output, label)
            except Exception:
                pdb.set_trace()
            #accuracy
            val_predicted = torch.argmax(output, dim = 1) 
            val_acc = torch.mean((val_predicted == label).float())
            val_prop_ones =  val_predicted.float().mean().detach()

            val_acc = torch.mean((val_predicted.squeeze() == label).float())
            val_losses.append(loss.detach())
            val_prop.append(val_prop_ones)
            val_accur.append(acc.detach())
    
    #pdb.set_trace()
    epoch_val_loss = torch.mean(torch.stack(val_losses))
    epoch_val_acc = torch.mean(torch.stack(val_accur))
    epoch_val_prop = torch.mean(torch.stack(val_prop))
    epoch_loss = torch.mean(torch.stack(losses))
    epoch_acc = torch.mean(torch.stack(accur))
    epoch_prop = torch.mean(torch.stack(prop))
    
    print("epoch:", epoch)
    print("prop:", epoch_prop)
    print("accuracy:", epoch_acc)
    print("loss:", epoch_loss)
    print("val_prop:", epoch_val_prop)
    print("val_accuracy:", epoch_val_acc)
    print("val_loss:", epoch_val_loss)
    writer.add_scalar("Accuracy/train", epoch_acc, epoch)
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/test", epoch_val_acc, epoch)
    writer.add_scalar("Loss/test", epoch_val_loss, epoch)

