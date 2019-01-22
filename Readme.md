# Perceptron for segregation of ATTACK and NONATTACK DDoS packets in a UDP traffic dataset

## Accuracy: 0.87

## The model
We're using a simple [perceptron](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-f01/www/handouts/110601.pdf), built with **Tensorflow.**

Although depth of the model is similar to that of a perceptron, the width of the model has been increased which lead us to better accuracy.
- Input: [ 4 nurons ]
- Hidden 1: [ 50 nurons ] FC (Fully Connected)
- Hidden 2: [ 50 nurons ] FC
- Hidden 3: [ 50 nurons ] FC
- Output: [ 2 nurons ] One-Hot

+ Weight initialization: RandomNormal
+ Activation:-
  + All Hidden: ReLU
  + Output: Softmax
+ Cost function: CrossEntropy
+ Optimizer: Adam (100 / 1,000,000 touple batch prop)

* 20 epochs of 10,000 iterations each


## Input
4 nodes: [SOURCE IP] [DESTINATION IP] [SOURCE PORT] [DESTINATION PORT]

> Touple in the input data set obviously match this format

> IP Address has been normalized. Please create `issues` suggesting IP normalization techniques.

Output: 0 / 1, for isAttackPacket === true.

## About the files
Contain about 1 million UDP packets.

> Please use Notepad++ or Sublime to view the data set on Windows

- raw_data_...txt: Attack and normal datasets taken from the internet
- IPTrain_data_...txt: Data and label for training the model
- test_data_...txt: Data and label for testing the trained model
- MODEL_TEST: Saved model