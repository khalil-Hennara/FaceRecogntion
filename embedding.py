#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("data", help="The data folder where you save your picture want to proccess bay attintion you should but each collection of feature in the Folder and the folder name will be the label to those faces.")
parser.add_argument("database",help="The folder where you want save the proccessing vector.")
args=parser.parse_args()


DATA_BASE=None
FOLDER_NAME=None
def usage():
    print("""Error: you have to provide a valid path for data folder where we have and proccessing image and database folder where we save the embbeding vector.
    """)

if (not args.data) or (not args.database):
      usage()
      sys.exit()

DATA_BASE=args.database
FOLDER_NAME=args.data

if not os.path.exists(DATA_BASE):
   os.makdir(os.path.join(os.getcwd(),DATA_BASE))

if not os.path.exists(FOLDER_NAME):
   usage()
   sys.exit()

print("import model.....")
from facenet_pytorch import MTCNN, InceptionResnetV1,extract_face
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import cv2 as cv
import time
import tqdm
from collections import defaultdict



def collate_fn(x):
    return x[0]

def embedding_data(folder_name,output_folder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    print("Build model...")
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )


    # Load facial recognition model
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    
    dataset = datasets.ImageFolder(folder_name)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn)

    aligned = []
    names = {}
    print("read image and Extract Faces.....")
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:6f}'.format(prob))
            aligned.append(x_aligned)
            index=len(aligned)-1
            if dataset.idx_to_class[y] in names:
                names[dataset.idx_to_class[y]].append(index)   
            else:
                names[dataset.idx_to_class[y]]=[index]
    if len(aligned)!=0:
       aligned = torch.stack(aligned).to(device)
       print("Encoding Face....")
       embeddings = resnet(aligned).detach().cpu()

       for name,value in names.items():
           print("Encoding {} Face...".format(name))
           avr=torch.sum(torch.Tensor([embeddings[k].numpy() for k in value]),axis=0)/len(value)
           torch.save(avr,"{}/{}.pkl".format(output_folder,name))
       return 1
       
    else:return 0
    
    
if __name__=="__main__":
    if embedding_data(FOLDER_NAME,DATA_BASE):
       print("Done")
       sys.exit(0)
    else:
       print("Somthing Error")
       sys.exit(1)

