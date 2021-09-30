#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("file", help="The video path you want process. ")
parser.add_argument("-o","--output", help="The file you want to save the process video default =[out] ")
args=parser.parse_args()

DATA_BASE=""
VIDEO=""
PROCEES_FILE="out.avi"

def usage():
     print("""
     	This program is to detect actor face from money heist TV show we can predict up to 11 person in the show.
     	to use this program you need to provide the database path where the embedding vector.
     	the path to video or the camera option
     """)
if (not args.data) or (not args.file):
      usage()
      sys.exit()

DATA_BASE=args.data
VIDEO=args.file
if args.output:
   PROCEES_FILE=args.output
   
print("Load Dependices....")
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


workers = 0 if os.name == 'nt' else 4




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


print("import model...")
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)


# Load facial recognition model
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()



def load_data(folder_name):
    names=[]
    data=[]
    for i in os.listdir(folder_name):
        names.append(i[:-4])
        vector=torch.load('{}/{}'.format(folder_name,i))
        data.append(vector)
    return names,data



cap=cv.VideoCapture(VIDEO)

names,data=load_data(DATA_BASE)

names=["None"]+names

v_len = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

font = cv.FONT_HERSHEY_SIMPLEX

out = cv.VideoWriter(PROCEES_FILE,cv.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))

time.sleep(2)

for _ in tqdm.tqdm(range(v_len)):
    
    _,frame=cap.read()
    
    rgb=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    
    faces, prob = mtcnn.detect(rgb)
    x_aligned=[]
    boxes=[]
    if faces is not None:
        for face in faces:
            x,y,w,h=face
            img=rgb[int(y):int(h),int(x):int(w)]
            try:
                tensor=mtcnn(img)
            except:
                pass
            if tensor is not None:
                x_aligned.append(tensor)
                boxes.append((int(x),int(y),int(w),int(h)))
    if len(x_aligned)!=0:     
        x_aligned = torch.stack(x_aligned).to(device)
        embeddings = resnet(x_aligned).detach().cpu()
        # predict=[np.argmin([(center-e1).norm().item() for center in data ]) for e1 in embeddings]
        predict=[]
        for e1 in embeddings:
            tmp=[]
            for center in data:
                tmp.append((center-e1).norm().item())
            if min(tmp)>1:
                predict.append(0)
            else: predict.append(np.argmin(tmp)+1)
        for index,(x,y,w,h) in enumerate(boxes):
            cv.rectangle(frame,(x,y),(w,h),(0,255,0))
            cv.putText(frame, str(names[predict[index]]), (x + 5, y - 5), font, 1, (255, 255, 0), 1)

    out.write(frame)
    key=cv.waitKey(1)&0XFF
    if key==ord('q'):
        cv.destroyAllWindows()
        break
cap.release()
out.release()    


