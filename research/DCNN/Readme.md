# Deep Convolutional Neural Network for Modelling Sentences
[Research Paper](https://arxiv.org/abs/1404.2188)


- Tensorflow 1.8
- Using new Dataset API for Input pipeline
- dcnn_model.py which contains implementation of DCNN research paper.
- Creating Serialized model (.pbtxt) for tensorflow serving which can be deployed in cloud ml

# Note

- Currently the code doesn't use pretrained embedding

# Instructions

- Download the dataset from [here](https://drive.google.com/open?id=1j9d1zyEaxVRwTm2zjOiQdJ5SxcusvZ_N)
- Extract the zip file to dataset folder.
- use build_vocab.py to generate vocabulary file
- train.py - Training the model

Accuracy = 0.84
===============
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/research/FastText/images/Accuracy.PNG)

Average Loss = 0.64
=====================
![alt text](https://github.com/KishoreKarunakaran/CloudML-Serving/blob/master/research/FastText/images/Loss.PNG)
