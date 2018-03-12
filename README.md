# CloudML Solutions

- How to build a serving model in Tensorflow ?
- How to use estimator effectively ?
- Serving the model in cloud ml
- Using Keras with Tensorflow Estimator
- How to serve scikit-learn models in Cloud-ML(WIP) ?
- Many More.......

The idea is to solve many real-world data problems using Machine Learning with Tensorflow. We will use various open dataset available from kaggle and other sources to solve the required problem using Tensorflow, scikit and deploying it in cloud ml.

This repository also contains implementation of research papers in tensorflow which can be deployed in cloud-ml

Feel free to support this repository by giving Star.

Contributions are welcome !!

# Deployment

To deploy the model in cloud-ml use the following command

### Tensorflow Model
gcloud ml-engine versions create {MODEL_VERSION} --model={MODEL_NAME} --origin={MODEL_PATH_IN_BUCKET}  --runtime-version=1.4

### Scikit-Learn
gcloud ml-engine versions create {MODEL_VERSION} --model={MODEL_NAME} --origin={MODEL_PATH(PICKLED FILE)} --runtime-version="1.4" --framework="SCIKIT_LEARN"

### Inference
gcloud ml-engine predict --model={MODEL_NAME} --version={MODEL_VERSION} --json-instances={INPUT_JSON_FILE}


# Research Papers Implementation

- [FastText](https://github.com/kishorenayar/Tensorflow-Solutions/tree/master/Research-Papers/FastText)



# Available Solutions

- [Kaggle Toxic Comment Multi Label Classification](https://github.com/KishoreKarunakaran/CloudML-Solutions/tree/master/text/ToxicComments_MultiLabel)
- [Kaggle consumer finance complaints - Based on Yoon Kim Research paper](https://github.com/KishoreKarunakaran/CloudML-Solutions/tree/master/text/finance-complaints)
- [Sms SpamClassification - Setting custom weights for imbalanced class, BOW model](https://github.com/KishoreKarunakaran/CloudML-Solutions/tree/master/text/SpamClassification)
- [Iris - Canned Estimator](https://github.com/KishoreKarunakaran/CloudML-Solutions/tree/master/structured-data/iris)
- [Mnist - Custom Estimator using Keras and more](https://github.com/KishoreKarunakaran/CloudML-Solutions/tree/master/structured-data/mnist)


<div align="center">
  <img src="https://github.com/kishorenayar/Tensorflow-Solutions/blob/master/extras/pics/Tensorflow-Scikit.jpg"><br><br>
  <img src="https://github.com/kishorenayar/Cloud-ML-Solutions/blob/master/extras/pics/image00.png"><br><br>
  <img src="https://github.com/kishorenayar/Cloud-ML-Solutions/blob/master/extras/pics/keras-logo-2018-large-1200.png"><br><br>
</div>
