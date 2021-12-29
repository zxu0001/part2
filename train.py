# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format= 'retina'

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
import json
from collections import OrderedDict

#import myfunction


#def main():
    
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(f'Current Mode in: {device}') 

#get train inputs
parser=argparse.ArgumentParser()
# Create  command line arguments using add_argument() from ArguementParser method

parser.add_argument('--data_dir',type=str, default='flowers', help='path to the folder of pet images') 
parser.add_argument('--save_dir',type=str, default='', help='path to the folder of checkpoint')
parser.add_argument('--arch', type=str, default='vgg19', help='CNN Model Architeture')
parser.add_argument('--gpu', type=str, default='gpu', help='Prefer training with GPU?')
parser.add_argument('--hiddenN', type=int, default=1000, help='# of hidden neurons')  
parser.add_argument('--epochs', type=int, default=15, help='# of epochs')  
parser.add_argument('--learningRate', type=float, default=0.001, help='learning rate')  

inargs=parser.parse_args()   
#trInput=get_input_args()

data_dir=inargs.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'  

tr_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

vt_transforms=transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFoldera
tr_data = datasets.ImageFolder(train_dir, transform=tr_transforms)
test_data = datasets.ImageFolder(test_dir,transform=vt_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=vt_transforms)  

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(tr_data,batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
validloader= torch.utils.data.DataLoader(valid_data, batch_size=32)
        
device=torch.device("cuda" if (torch.cuda.is_available() and inargs.gpu =='gpu') else "cpu")

#flower disctionary
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    #print(type(cat_to_name))
    #cat_to_name['1']

#construct model, with 1 hidden layer, load if choose vgg19 

hidsize=inargs.hiddenN
learning_rate=inargs.learningRate

droprate=0.4

if inarg.arc=='vgg19':
    model=models.vgg19(pretrained=True)
    
    for param in model.parameters():
    param.requires_grad = False 
    
    #from collections import OrderedDict    
    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088,hidsize)),
    ('relu', nn.ReLU()),
    ('dropout1',nn.Dropout(droprate)),
    ('fc2', nn.Linear(hidsize,102)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
    
elif inarg.arc=='densenet121':
    model=models.densenet121(pretrained=True)
    
    for param in model.parameters():
    param.requires_grad = False 
    
    #from collections import OrderedDict    
    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024,hidsize)),
    ('relu', nn.ReLU()),
    ('dropout1',nn.Dropout(droprate)),
    ('fc2', nn.Linear(hidsize,102)),
    ('output', nn.LogSoftmax(dim=1))
    ]))


model.classifier = classifier

criterion=nn.NLLLoss()

optimizer=optim.Adam(model.classifier.parameters(), lr=learning_rate)

#device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Current device: {device}')
model.to(device)

epochs=15
steps=0
runloss=0
printstep=50

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps+=1
        
        inputs, labels = inputs.to(device),labels.to(device)
        
        optimizer.zero_grad()
        
        lgps=model.forward(inputs)
        loss=criterion(lgps,labels)
        loss.backward()
        optimizer.step()
        
        runloss += loss.item()
        
        if steps % printstep ==0:
            validloss=0
            accuracy=0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels =inputs.to(device), labels.to(device)
                    lgps=model.forward(inputs)
                    vloss=criterion(lgps, labels)
                    
                    validloss +=vloss.item()
                    
                    ps=torch.exp(lgps)
                    topp, topc =ps.topk(1,dim=1)
                    equals= topc == labels.view(*topc.shape)
                    accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f'Epoch: {epoch+1} .{epochs}. '
                  f'Train Loss: {runloss/printstep:.3f}. '
                  f'Test Loss: {validloss/len(validloader):.3f}. '
                  f'Test Accuracy: {accuracy/len(validloader):.3f}. ')
            runloss=0
            model.train()
            


model.to('cpu')

model.class_to_idx = tr_data.class_to_idx

print("Trained model: \n\n",model, '\n')
print("The state dic keys: \n\n", model.state_dict().keys())

checkpoint= {'arch': inargs.arch,
             'classifier': model.classifier,
            'learning_rate': learning_rate,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx,
            'optimizer_dict':optimizer.state_dict()}

if inargs.save_dir!='':
    torch.save(checkpoint, inargs.save_dir+'/checkp.pth')
else:
    torch.save(checkpoint, 'checkp.pth') 


    
    


        
    


