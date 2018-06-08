import numpy as np
import os

'''
BaseLine model - Using standard 213 Class dataset
	classes - 213
	total videos - 1520
	test - 213
	val - 213
	train - 1094
	expected accuracy - 1 / 213
'''

# Overall - avg on whole datset
accuracy  = 0.0

for i in range(1000):
	data = "../data_final"
	folders = os.listdir(data)

	label_count = 0
	correct = 0
	total = 0

	for label in folders:
		if label ==".DS_Store":
			continue

		cur_class_folder = os.path.join(data, label)
		for num_vid in range(len(os.listdir(cur_class_folder))):
			if(np.random.randint(0, 213) == label_count):
				correct += 1
			total += 1

	accuracy += correct*1.0/total
print(accuracy/1000.0)


print("total videos: " + str(total))
print("correct: " + str(correct))
print("accuracy: " + str(correct*1.0/total))

# theoretical testing on train, test, val - just changing number of iterations
# correct = 0
# for i in range(213):
# 	guess = np.random.randint(0, 213)
# 	if(guess == i):
# 		correct += 1
# print("avg: " + str(correct*1.0/213))
