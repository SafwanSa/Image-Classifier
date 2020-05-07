# Imports here
import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
import torch.nn.functional as F
from workspace_utils import active_session
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




def build(model_arch, n_hidden, device):
    model = models.vgg13(pretrained=True)
    n_inputs = 25088
    if model_arch == "densenet121":
        model = models.densenet121(pretrained=True)
        n_inputs = 1024
        
    for param in model.parameters():
        param.requires_grad = False
      
    classifier = nn.Sequential(nn.Linear(n_inputs, n_hidden),
                              nn.ReLU(),
                              nn.Dropout(0.25),
                              nn.Linear(n_hidden,102),
                              nn.LogSoftmax(dim=1))
    model.classifier = classifier
    model.to(device)
    print("Building Finished ...")
    return model



def train(model, epochs, device, dataloaders, learning_rate, criterion, optimizer):
    steps = 0
    print_time = 5
    running_loss = 0
    trainloader = dataloaders[0]
    validloader = dataloaders[1]
    for e in range(epochs):
        for images, labels in trainloader:
            steps+=1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            
            
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()

            if steps % print_time == 0:
                model.eval()
                acc = 0
                valid_loss = 0
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)

                        log_ps = model.forward(images)
                        loss = criterion(log_ps, labels)
                        
                        valid_loss+=loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        acc += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_time:.3f}.. "
                      f"Test loss: {valid_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {acc*100/len(validloader):.3f}")
                running_loss = 0
                model.train()
     
    print("Training Finished ...")


    
def predict(image, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    with torch.no_grad():
        model.eval()
        image = image.unsqueeze_(0)#becasue of the missing batch size
        image = image.float()#becasue it is double
        log_ps = model.forward(image.to(device))
        ps = torch.exp(log_ps)
        return ps.topk(topk, dim=1)
