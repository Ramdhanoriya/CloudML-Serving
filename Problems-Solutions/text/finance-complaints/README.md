# Kaggle Consumer Finance Complaints Classification

More on here ![alt text](https://www.kaggle.com/cfpb/us-consumer-finance-complaints)

- Tensorflow 1.4
- Using new Dataset API for Input pipeline
- Creating Serialized model (.pbtxt) for tensorflow serving which can be deployed in cloud ml
- Local prediction using serialized file

# Instructions

- use build_vocab.py to generate vocabulary file
- train.py - Training the model
- inference.py - model inference
- cnn_model.py - Based on Convolutional Neural Networks for Sentence Classification - Yoon Kim Research paper.

Accuracy = 0.92
===============
![alt text](https://github.com/kishorenayar/Machine-Learning-Solutions/blob/master/Problems-Solutions/text/finance-complaints/images/Accuracy.PNG)

Average Loss = 0.5
=====================
![alt text](https://github.com/kishorenayar/Machine-Learning-Solutions/blob/master/Problems-Solutions/text/finance-complaints/images/Loss.PNG)
