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
import json

def transform_data():
    train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                     transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    print("Data Transforms created ...")
    return [train_transforms, valid_transforms, test_transforms]


def create_loaders(datasets):
    
    trainloader = torch.utils.data.DataLoader(dataset = datasets[0], batch_size = 32, shuffle= True)
    validloader = torch.utils.data.DataLoader(dataset = datasets[1], batch_size = 32)
    testloader = torch.utils.data.DataLoader(dataset = datasets[2], batch_size = 32)
    print("Dataloaders created ...")
    return [trainloader, validloader, testloader]

def load_datasets(data_dir):
    data_transforms = transform_data()
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_dataset = datasets.ImageFolder(train_dir, transform = data_transforms[0])
    valid_dataset = datasets.ImageFolder(valid_dir, transform = data_transforms[1])
    test_dataset = datasets.ImageFolder(test_dir, transform = data_transforms[2])
    print("Datasets loaded ...")
    return [train_dataset, valid_dataset, test_dataset]



def save_checkpoints(save_dir, model, optimizer, epochs, learning_rate, train_dataset):
    model.class_to_idx = train_dataset.class_to_idx
    chkpoint = {
        'model':model,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epochs': epochs,
        'learning_rate': learning_rate
    }
    torch.save(chkpoint, save_dir)
    print("Checkpoint saved succesfully...")
    

def load_checkpoint(load_dir):
    checkpoint = torch.load(load_dir)
    print("Checkpoint loaded succesfully...")
    return checkpoint

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((10000, 256))
    else:
        pil_image.thumbnail((256, 10000))

    
    l = (pil_image.width-224)/2
    b = (pil_image.height-224)/2
    r = l + 224
    t = b + 224
    pil_image = pil_image.crop((l,b,r,t))
    
    
    np_image = np.array(pil_image)/255
    np_mean = np.array([0.485, 0.456, 0.406])
    np_std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image-np_mean)/np_std             
    np_image = np_image.transpose([2,0,1])
    return torch.from_numpy(np_image)


def load_categories(cat_dir):
    with open(cat_dir, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name