import os
import label_image
import argparse

import numpy as np
import tensorflow as tf

'''
	This code adapted from https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py

	Run retrained CNN on test videos
'''

def testModel(pathToFrames, modelFile, labelFile, inputLayer, outputLayer):

	graph = label_image.load_graph(modelFile)
	correct = 0
	total = 0

	for label in os.listdir(pathToFrames):
		if label == '.DS_Store':
			continue

		cur_class = os.path.join(pathToFrames, label)
		classified = {}

		for frame in os.listdir(cur_class):
			if(label+"_0" in frame):

				cur_frame = os.path.join(cur_class, frame)

				# run model on this frame
				t = label_image.read_tensor_from_image_file(cur_frame)

				input_name = "import/" + inputLayer
				output_name = "import/" + outputLayer
				input_operation = graph.get_operation_by_name(input_name)
				output_operation = graph.get_operation_by_name(output_name)

				with tf.Session(graph=graph) as sess:
					results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
				results = np.squeeze(results)
				top_k = results.argsort()[-3:][::-1]
				labels = label_image.load_labels(labelFile)
				# print(top_k)
				for i in top_k:
					if labels[i] in classified:
						classified[labels[i]] += 1
					else:
						classified[labels[i]] = 1
					break
		best = keywithmaxval(classified)
		label_clean = convertLabel(label)
		print(best)
		print(label_clean)
		if(best == label_clean):
			correct += 1
		total += 1
		print(total)
		print(correct)

	return correct*1.0/total

# return key with max value in dict d
def keywithmaxval(d):
	v=list(d.values())
	k=list(d.keys())
	return k[v.index(max(v))]

def convertLabel(label):
	label = label.lower()
	split = label.split('-')
	combine = ""
	for i, segs in enumerate(split):
		if(i == len(split) - 1):
			combine += segs
		else:
			combine += segs + " "
	return combine





