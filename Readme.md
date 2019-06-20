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

> Input for `input tensor layer` matches the below format:

4 nodes: [SOURCE IP] [DESTINATION IP] [SOURCE PORT] [DESTINATION PORT]

[SOURCE IP, DESTINATION IP] have been normalized:
- Actual IP addresses have been obfuscated by the source dataset
- Renormalization has been done to be able to train a model on IP addresses:
`w * 1 + x * 2 + y * 3 + z * 4`, where `w.x.y.z` is an IP

> Recommend you to develop a better normalization technique for IP address.

Output: 0 / 1, for isAttackPacket === true.

## About files

Dataset: 1 million UDP packets

Files inside /Data\ Sets are self-explanatory.
MODEL_TEST contains saved model.

# License

```
The MIT License (MIT)

Copyright 2019 Dhruvaraj nagarajan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```