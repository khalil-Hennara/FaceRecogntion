# Face Recognotion 
In this project I create  face recognition app where I can add,delete,edit faces in my database.
I use pretrained FaceNet model to encoding face in vector of length 512.
this Network is traning on alot of faces in such way, that can represent the face as embedding vector which represent the information of the face.
I use pretrained model you can see the details on this repesotory link [FaceNet] (https://github.com/timesler/facenet-pytorch)

to train model to recognice faces or to repreent faces as vectors we can use two way the first one as DeepFace on this paper[DeepFace: Closing the Gap to Human-Level Performance in Face Verification](https://ieeexplore.ieee.org/document/6909616) the second one is the faceNet from this paper [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
becouse I use the FaceNet I will explaie a litel.

lits Asume I have three picture as follow 

<div>
    <br/>
    <img src='Image_1.png' style="float:left" >
    <img src='Positive.png' style="float:left;margin-left:200px;">
    <img src='Negative.png' >
</div>
<div>
 <p style="float:left">Image</p><p style="float:left;margin-left:300px">Positive</p><p style="margin-left:650px">Negative</p>
    <br/>
    <br/>
</div>



I want the embedding vector how represent the image to be as same as posible from the positive one and as fare as posible for  the negative one so we will use this form 

$||f(image)-f(positive)||^2<=||f(image)-f(Negative)||^2$

**or**

$||f(image)-f(positive)||^2-||f(image)-f(Negative)||^2<=0$

the $f$ is the function the model learn to encoding the images,but with this formela if the simple soulation is to represent all image by the **same** vector or the **zero** vector.

so instaed we will add hipyerparameter to the last equation to become as follow.

$||f(image)-f(positive)||^2+epsilon<=||f(image)-f(Negative)||^2$

in this way, we force the model to make the gap between positiv and negative as big as posible.

so we can deffine the **loss function** as follow.

$l(I,P,N)=||f(I)-f(P)||^2+epsilon-||f(I)-f(N)||^2$
and then $L=max(l(I,P,N),0)$

the cost is the sum over all traning example 
$cost=∑_i(L(I_i,P_i,N_i)$

and that all you need to train a model to regoneation faces or you can use a pretrained model which will save your time.

in This project as I mention first I use pretraind model from pytorch framework and make an application the take any amount of picture and represent one vector with 512 dimisnion.

## How it's work

the model take a photo and detect how many faces in this photo then encoding this faces into vector with 512 dimension.
but this will be bad for traning so to make the model work will first you need to make each photo have one face exactly then but all photo for the same person into one folder and give this folder the Id you want for that person.
but all folders (the folder containe the faces) into one folder and then pass the path of this folder to the embedding script.
this script load the model and then read faces from each folders and then pass this image to the model which output 512 for each face but what i did I toke all vectors for each persone and take the avreage of this vectors to represent the final vector for one persone and this process continue for each person in the data.
example of using embedding.py


```python
$python embedding.py data database
```

where data is the folder containe the images and database where the embedding vector will be save.

whith this structure you can add faces to the database or delete faces from the database it's very efishaint process.

the second step is to call the **face_recognition** file which take the path of the database and the path of the video you want to process you can even use life camera by modify one line in the code it's very easy step.
I use video becouse i process the video on colab but as I say you can use any kind of straem.

the model will run until the video finished or stop the stream by taping ctrl+c

example:


```python
$python face_recognition.py database videopath outputpath 
```

*database* is where we save the representation vector the *videopath* is the path of video the *outputpath* is where you want to save the processed video.

### Testing 
for testing, I use from 1-10 image for 11 actors from Money heist TV series to recognize them(Prof,Rio,Tokyo,Gundia,Bogota,Denver,Arturo,Nairobi,Monica,Helsinki,Berlin),embedding this image for each person in vectors and then take the average over this vector,creating one vector that represent one person.
this Technique achieve a good accuracy as we can see in demo, and by using ecludian distance as metrics get the best results.
in This app you can add face to the data or delete face without any retraining and also you can edit faces by adding more image to the training data.
in this link we see another video where I add Lisbon


```python

```
