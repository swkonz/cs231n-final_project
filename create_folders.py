import os
from shutil import copy

def create_folders(directory, trainDir = "processed/train/", testDir="processed/test/"):

	train_dir, test_dir = trainDir, testDir
	os.makedirs(trainDir)
	os.makedirs(testDir)

	for class_dir in os.listdir(directory):
		if class_dir == ".DS_Store":
			continue

		class_train_dir_path = os.path.join(train_dir, class_dir)
		class_test_dir_path = os.path.join(test_dir, class_dir)
		os.makedirs(class_train_dir_path)
		os.makedirs(class_test_dir_path)

		cur_class_path = os.path.join(directory, class_dir)

		count = 0
		for vid in os.listdir(cur_class_path):
			cur_vid_path = os.path.join(cur_class_path, vid)
			if(count == 0): # move to test
				copy(cur_vid_path, class_test_dir_path)
			else:
				copy(cur_vid_path, class_train_dir_path)
			count += 1