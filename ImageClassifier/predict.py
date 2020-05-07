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
    
    parser.add_argument('image_dir', action="store", default="/home/workspace/ImageClassifier/flowers/test/10/image_07090.jpg")
    parser.add_argument('checkpoint', action="store", default="/home/workspace/ImageClassifier/checkpoint.pth")
    parser.add_argument('--top_k', action="store",dest="top_k",type=int, default=5)
    parser.add_argument('--category_names', action="store",dest="classes_names", default="/home/workspace/ImageClassifier/cat_to_name.json")
    parser.add_argument('--gpu', action="store",dest="device", default="gpu")

    p_args = parser.parse_args()
    
    #gpu or cpu
    device = torch.device("cpu")
    if torch.cuda.is_available() and p_args.device == 'gpu':
        device = torch.device("cuda:0")
    #load the categories names
    cat_names = ut.load_categories(p_args.classes_names)
    #load the checkpoint
    chkpoint = ut.load_checkpoint(p_args.checkpoint)
    #configure the model
    model = chkpoint['model']
    model.load_state_dict(chkpoint['model_state'])
    #process the image
    image = ut.process_image(p_args.image_dir)
    #predict
    probs, classes = brain.predict(image, model, p_args.top_k, device)
    
    #configure the probabilities and class names
    probs = probs.cpu().numpy().flatten().tolist()
    classes = classes.cpu().numpy().flatten() .tolist()
    index_to_class = {value: key for key, value in model.class_to_idx.items()}
    print("\n\n\n\n{:30s}{:20s}".format("Class Name"," Probability"))
    for i in range(len(classes)):
        name = [cat_names[index_to_class[classes[i]]]][0]
        probability = probs[i]*100
        print("{:30s}{:0.3f}".format(name, probability))
    
if __name__ == "__main__":
    main()
