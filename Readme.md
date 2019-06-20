# Perceptron for classification of DDoS UDP traffic

Passively categorize UDP dataset into attack and non-attack datasets.

# The model

## Accuracy: 0.87

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


# Input
4 nodes: [SOURCE IP] [DESTINATION IP] [SOURCE PORT] [DESTINATION PORT]

> Touple in the input data set obviously match this format

> IP Address has been normalized.

Output: 0 / 1, for isAttackPacket === true.

## About the files
Contain about 1 million UDP packets.

> Please use Notepad++ or Sublime to view the data set on Windows

- raw_data_...txt: Attack and normal datasets taken from the internet
- IPTrain_data_...txt: Data and label for training the model
- test_data_...txt: Data and label for testing the trained model
- MODEL_TEST: Saved model

# License

```
The MIT License (MIT)

Copyright 2019 Dhruvaraj nagarajan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```