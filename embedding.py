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
    names = []
    print("read image and Extract Faces.....")
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])
    if len(aligned)!=0:
       aligned = torch.stack(aligned).to(device)
       print("Encoding Face....")
       embeddings = resnet(aligned).detach().cpu()
       avr=torch.sum(embeddings,axis=0)/len(embeddings)
       torch.save(avr,"{}/{}.pkl".format(output_folder,names[0]))
       return 1
       
    else:return 0
    
    

