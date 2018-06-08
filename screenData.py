import os
import sys
import shutil

'''
Script to remove all directories in dataset that have fewer than 4 videos in them
'''

pathToVids = "/media/sean/My Passport/data"
files = os.listdir(pathToVids)

names = set()

for filename in files:
        
        if(filename == '.DS_Store'):
            continue

        cur = os.path.join(pathToVids, filename)

        # check if a duplicate
        if (filename not in names):
        	# add it
        	names.add(filename)
        else:
        	shutil.rmtree(cur)

        # check if > 4 videos
        if (len(os.listdir(cur)) < 4):
        	shutil.rmtree(cur)

# finished going through all data
print(len(names))
print(names)


