import argparse
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
import utilities as ut
import brain
parser = argparse.ArgumentParser()


def main():
    
    parser.add_argument('data_dir', action="store", default="/home/workspace/ImageClassifier/flowers")
    parser.add_argument('--save_dir', action="store",dest="save_dir", default="/home/workspace/ImageClassifier/checkpoint.pth")
    parser.add_argument('--arch', action="store",dest="model_arch", default="vgg13")
    parser.add_argument('--learning_rate', action="store",dest="learning_rate", type=float, default=0.001)
    parser.add_argument('--epochs', action="store",dest="epochs",type=int, default=1)
    parser.add_argument('--hidden_units', action="store",dest="n_hidden", type=int, default=500)
    parser.add_argument('--gpu', action="store",dest="device", default="gpu")
    
    #get the parser arguments
    p_args = parser.parse_args()
    learning_rate = p_args.learning_rate
    epochs = p_args.epochs
    #gpu or cpu
    device = torch.device("cpu")
    if torch.cuda.is_available() and p_args.device == 'gpu':
        device = torch.device("cuda:0")

    #training and proeccing funcstions
    #load the datasets and get the dataloaders
    datasets = ut.load_datasets(p_args.data_dir)
    dataloaders = ut.create_loaders(datasets)
    #buld the model
    model = brain.build(p_args.model_arch, p_args.n_hidden, device)
    #setup the loss fucntions and the optimizer agent
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    #train the model
    brain.train(model, epochs, device, dataloaders, learning_rate, criterion, optimizer)
    #save the model
    ut.save_checkpoints(p_args.save_dir, model, optimizer, epochs, learning_rate, datasets[0])
    
if __name__ == "__main__":
    main()
