import cv2
import os
import sys
import shutil


pathToVids = "data_final/"
pathToFrames = "frames_final/"

folders = os.listdir(pathToVids)

for name in folders:
        
    if(name == '.DS_Store'):
        continue

    # create new folder for frames
    cur_class_folder = os.path.join(pathToFrames, name)
    os.makedirs(cur_class_folder)

    # move into this class folder
    cur_class = os.path.join(pathToVids, name)
    vids = os.listdir(cur_class)

    # count = 0
    actual_vid = 0

    # loop through all vids
    for vid in vids:
    	# vid to frames and move to new folder
        cur_vid = os.path.join(cur_class, vid)
        cur_cap = cv2.VideoCapture(cur_vid)
        target = os.path.join(cur_class_folder, name)
        
        success, frame = cur_cap.read()
        frame_count = 0
        true_count = 0
        success = True
        while success:
            #save frame to file as jpg -- only save after the 20th frame since all the videos start off slow
            if(frame_count >= 20 and frame_count % 4 == 0):
                cv2.imwrite(target+"_%d_%d.jpg" % (actual_vid, true_count), frame)
                true_count += 1

            frame_count += 1
            success, frame = cur_cap.read()
        actual_vid += 1


