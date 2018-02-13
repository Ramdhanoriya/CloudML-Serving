# FastText Implementation using Tensorflow

[Research Paper](https://arxiv.org/abs/1607.01759)


- Tensorflow 1.4
- Using new Dataset API for Input pipeline
- FastTextEstimator which contains implementation of fasttext research paper.
- Creating Serialized model (.pbtxt) for tensorflow serving which can be deployed in cloud ml
- Local prediction using serialized file

# Instructions

- Download the dataset from [here](https://drive.google.com/open?id=1j9d1zyEaxVRwTm2zjOiQdJ5SxcusvZ_N)
- Extract the zip file to dataset folder.
- use build_vocab.py to generate vocabulary file
- train.py - Training the model

Accuracy = 0.92
===============
![alt text](https://github.com/kishorenayar/Tensorflow-Solutions/blob/master/Research-Papers/FastText/images/Accuracy.PNG)

Average Loss = 0.22
=====================
![alt text](https://github.com/kishorenayar/Tensorflow-Solutions/blob/master/Research-Papers/FastText/images/Loss.PNG)
