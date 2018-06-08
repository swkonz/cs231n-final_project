import numpy as np
import os
import cv2
import pickle
from shutil import copyfile

'''
	LoadData(datafolder)
	input: 
		datafolder - container directory for entire dataset
'''
def loadData(dataFolder):
	images = []

	# get all images names
	for root, dirs, files in os.walk(dataFolder):

		for file in files:
			if file.endswith('.jpg'):
				images.append(file)

	num_images = len(images)

	# set labels
	labeled_images = []
	for image in images:

		# get id tag
		temp_id = image.replace('.jpg', '').split('_')
		identifier = "_" + temp_id[1] + "_" + temp_id[2]

		label = temp_id[0]

		# add to list
		labeled_images.append([image, label])

	with open('data/labeled-frames.pkl', 'wb') as fout:
		pickle.dump(labeled_images, fout)

	return labeled_images

def extractFrames(data, target):

	for name in os.listdir(data):
		if(name == '.DS_Store'):
			continue

		cur_class_folder = os.path.join(data, name)

		for frame in os.listdir(cur_class_folder):
			# target
			cur_frame_target = os.path.join(target, frame)
			# src
			cur_frame_src = os.path.join(cur_class_folder, frame)

			# copy frame to source
			copyfile(cur_frame_src, cur_frame_target)








