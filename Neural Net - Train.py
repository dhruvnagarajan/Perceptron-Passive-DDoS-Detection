'''
The MIT License (MIT)

Copyright 2019 Dhruvaraj nagarajan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import tensorflow as tf
import numpy as np
from plotly.offline import plot
import plotly.graph_objs as ob
raw_data = open ("IPtrain_data_udp.txt"). readlines ()
raw_labels = open ("IPtrain_label_udp.txt"). readlines ()

x = []
y = []
gra=[]
gro=[]

for _ in raw_data:

	_ = _.split ()
	x.append ([int (_[0]), int (_[1]), int (_[2]), int (_[3])])

for _ in raw_labels:

	y.append ([0, 1])


def get_next_batch (batchSize):

	_data = raw_data[get_next_batch.counter : get_next_batch.counter + batchSize]
	_label = raw_labels[get_next_batch.counter : get_next_batch.counter + batchSize]

	get_next_batch.counter += batchSize

	batch_data = []
	batch_label = []

	for _ in _data:

		_ = _.split ()
		batch_data.append ([_[0] ,_[1], _[2], _[3]])
	
	for _ in _label:

		_ = _.split ()
		batch_label.append ([_[0], _[1]])

	return np.array (batch_data), np.array(batch_label)

nodesForLayerInput = 4
nodesForLayer1 = 50
nodesForLayer2 = 50
nodesForLayer3 = 50
nodesForLayerOut = 1

numberOfClassesOut = 2

data = tf.placeholder ('float', shape = [None, 4])
label = tf.placeholder ('float')

layer1 = {
		
		'w' : tf.Variable (tf.random_normal ([4, nodesForLayer1])),
		'b' : tf.Variable (tf.random_normal ([nodesForLayer1]))
	}

layer2 = {
	
	'w' : tf.Variable (tf.random_normal ([nodesForLayer1, nodesForLayer2])),
	'b' : tf.Variable (tf.random_normal ([nodesForLayer2]))
}

layer3 = {
	
	'w' : tf.Variable (tf.random_normal ([nodesForLayer2, nodesForLayer3])),
	'b' : tf.Variable (tf.random_normal ([nodesForLayer3]))
}

layerOut = {
	
	'w' : tf.Variable (tf.random_normal ([nodesForLayer3, numberOfClassesOut])),
	'b' : tf.Variable (tf.random_normal ([numberOfClassesOut]))
}

saver = tf.train.Saver ()

def graph (_data):

	ansLayer1 = tf.nn.relu (tf.add(tf.matmul(_data, layer1['w']), layer1['b']))
	ansLayer2 = tf.nn.relu (tf.add(tf.matmul(ansLayer1, layer2['w']), layer2['b']))
	ansLayer3 = tf.nn.relu (tf.add(tf.matmul(ansLayer2, layer3['w']), layer3['b']))

	ansLayerOut = tf.add(tf.matmul(ansLayer3, layerOut['w']), layerOut['b'])

	return ansLayerOut

def train (_x):

	prediction = graph (_x)

	cost = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits (

		_sentinel = None,
		logits = prediction,
		labels = label,
		dim = -1,
		name = None)
	)

	optimiser = tf.train.AdamOptimizer ().minimize (cost)

	nEpochs = 20

	with tf.Session () as sess:

		sess.run (tf.global_variables_initializer ())

		for epoch in range (nEpochs):

			epoch_loss = 0
			test=0
			get_next_batch.counter = 0

			for i in range (10000):

				epoch_data, epoch_label = get_next_batch (100)

				i, c = sess.run ([optimiser, cost], feed_dict = {data : epoch_data, label : epoch_label})

				epoch_loss += c
			
			print ("Training Batch: of Epoch: " + str (epoch + 1) + "\tLoss: " + str (epoch_loss))
			
			gra.append(epoch_loss)
			
		save_path = saver.save (sess, "/dir/model_train.ckpt")

		print ("Saved to: ", save_path)

		correct = tf.equal (tf.argmax (prediction, 1), tf.argmax (label, 1))

		accuracy = tf.reduce_mean (tf.cast (correct, 'float'))

		print ("Accuracy ", accuracy.eval ({data : x, label : y}))

train (data)

gra1 = list (range (len (gra)))

trace0 = ob.Scatter (
	
	x = gra1,
	y = gra,
	mode = 'lines+markers'
	)

data = [trace0]

plot (data,filename = "graph.html")
