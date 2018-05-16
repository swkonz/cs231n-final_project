import os
import numpy as np
import cv2

'''
Currently written to test for the largest height and width of a video
'''
def gatherDataAsArray(root):
    # X = np.empty((0, 2020, 480, 640))
    # y = np.empty((0, 1))
    X = []
    largest_width = 0
    largest_height = 0
    for path, subdirs, files in os.walk(root):
        for vid in files:
            # X.append(convertVidToArray(path + '/' + vid))
            h, w = convertVidToArray(path + '/' + vid)
            largest_height = h if (h > largest_height) else largest_height
            largest_width = w if (w > largest_width) else largest_width

    print(largest_width, largest_height)
    return X

def convertVidToArray(vid):
    vidcap = cv2.VideoCapture(vid)
    count = 0
    success = True
    buf = np.empty((2020, 656, 280, 3), np.dtype('uint8'))
    largest_width = 0
    largest_height = 0
    while success:
      # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
      success, image = vidcap.read()
      if (success):
          # buf[count] = image
          largest_height = image.shape[0] if (image.shape[0] > largest_height) else largest_height
          largest_width = image.shape[1] if (image.shape[1] > largest_width) else largest_width
          count += 1

    return largest_width, largest_height
