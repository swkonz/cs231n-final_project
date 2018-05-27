import numpy as np
import cv2
import os
import shutil
from skimage import filters
from skimage import transform

"""
Function: move
==============
Moves file from source to destination.
==============
input:
    source: where the file is currently at
    destination: where we want the file to be
output:
    None
"""
def move(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    shutil.move(source, destination + "/" + source)

"""
Function: computeSobelMag
=========================
Computes the magnitude using the sobel gradient.
=========================
input:
    frame: input image
output:
    mag: sobel magnitude of frame
"""
def computeSobelMag(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the Sobel gradient magnitude
    mag = filters.sobel(gray.astype("float"))

    return mag

"""
Function: computeSobelMagOfVid
==============================
Takes a video array and computes summation of sobel magnitude for each frame.
==============================
input:
    vid_array: array of video of shape (F, H, W, 3)
output:
    mag: summation of sobel magnitudes for each frame
"""
def computeSobelMagOfVid(vid_array):
    mag = np.zeros((vid_array.shape[1], vid_array.shape[2])).astype("float")

    # Compute Sobel mag for each frame and sum
    for frame in range(vid_array.shape[0]):
        mag += computeSobelMag(vid_array[frame])

    return mag

"""
Function: removeSeam
====================
Removes a single seam from the input video in the direction give.
====================
input:
    vid_array: array of video of shape (F, H, W, 3)
    dir: direction to remove seam, either 'vertical' or 'horizontal'
    mag: magnitude from computeSobelMag
output:
    new_vid_array: new video with seam removed from each frame
"""
def removeSeam(vid_array, dir, mag):
    F, H, W, _ = vid_array.shape

    newH = H - 1 if (dir == 'horizontal') else H
    newW = W - 1 if (dir == 'vertical') else W

    new_vid_array = np.zeros((F, newH, newW, 3)).astype(np.uint8)

    for frame in range(F):
        new_frame = transform.seam_carve(vid_array[frame], mag, dir, 1) * 255.0
        new_vid_array[frame] = new_frame.astype(np.uint8)

    return new_vid_array.astype(np.uint8)

"""
Function: resolution
====================
Takes a video and trims the height and width of each frame to be of shape H x W.
====================
input:
    vid_array: array of video of shape (F, oldH, oldW, 3)
    H: height each of the frames of the video should be reshaped to
    W: width each of the frames of the video should be reshaped to
output:
    new_vid: video where each frame is of shape H x W
"""
def resolution(vid_array, H, W):
    # Compute number of seams to remove in horizontal and vertical axes
    h_to_cut = vid_array.shape[1] - H
    w_to_cut = vid_array.shape[2] - W

    # Compute the maximum number of cuts needed
    max_cut = h_to_cut if (h_to_cut > w_to_cut) else w_to_cut

    new_vid = np.copy(vid_array)

    # Perform seam carving on video: vertical then horizontal
    for cut in range(max_cut):
        if (cut < w_to_cut):
            mag = computeSobelMagOfVid(new_vid)
            new_vid = removeSeam(new_vid, 'vertical', mag)
        if (cut < h_to_cut):
            mag = computeSobelMagOfVid(new_vid)
            new_vid = removeSeam(new_vid, 'horizontal', mag)

    return new_vid.astype(np.uint8)

"""
Function: getData
=================
Gets the number of frames, fps, height and width of video.
=================
input:
    cap: VideoCapture of a video from dataset
output:
    frames: number of frames of video
    fps: fps of video
    height: height of video
    width: width of video
"""
def getData(cap):
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    return frames, fps, height, width

"""
Function: findMinData
=====================
Finds the minimum number of frames, fps, height and width of all videos.
=====================
input:
    path_to_vids: path to where the videos are located, where each set of videos
                  are in there own folder
output:
    N_frames: smallest number of frames of all videos
    FPS: smallest fps of all videos
    H: smallest height of all videos
    W: smallest width of all videos
"""
def findMinData(path_to_vids):
    # Set to -1 before finding lowest values for each
    N_frames, FPS, H, W = -1, -1, -1, -1

    # Search through all videos to find smallest frames, height, and width
    for path, subdirs, files in os.walk(path_to_vids):
        for vid in files:
            if (vid == ".DS_Store"):
                continue

            cap = cv2.VideoCapture(path + '/' + vid)

            if (cap.isOpened()):
                # Get number of frames, height, and width
                frames, fps, height, width = getData(cap)

                # Update N_frames, FPS, H, and W
                N_frames = frames if (N_frames == -1 or frames < N_frames) else N_frames
                FPS = fps if (FPS == -1 or fps < FPS) else FPS
                H = height if (H == -1 or height < H) else H
                W = width if (W == -1 or width < W) else W

    return N_frames, FPS, H, W

"""
Function: convertVidToArray
===========================
Converts the VideoCapture object data to an array of shape (F, H, W, 3).
===========================
input:
    path_to_vid: path to where the video is located
    N_frames: number of frames the video should be
output:
    vid: an array representation of a VideoCapture object
"""
def convertVidToArray(path_to_vid, N_frames):
    cap = cv2.VideoCapture(path_to_vid)

    # Get number of frames, height, and width
    frames, fps, height, width = getData(cap)

    # Calculate the number of frames to cut from front and back
    frames_to_cut = 0 if (frames == N_frames) else frames - N_frames

    vid = np.zeros((frames - frames_to_cut, height, width, 3))

    if (cap.isOpened()):
        cur_frame = 0
        while (cur_frame < vid.shape[0]):
            ret, frame = cap.read()

            if (ret):
                vid[cur_frame] = frame
                cur_frame += 1
            else:
                break

    cap.release()

    return vid.astype(np.uint8)

"""
Function: preprocess
====================
Takes the path to a set of videos and makes each video contain the same number
of frames, same fps, and the same resolution. Saves the videos to a folder
called preprocessed_data.
====================
input:
    path_to_vids: path to where the videos are located, where each set of videos
                  are in there own folder
output:
    None
"""
def preprocess(path_to_vids):
    # Find minimum N_frames, FPS, H, and W
    N_frames, FPS, H, W = findMinData(path_to_vids)

    count = 0

    # Cycle through all videos trimming the frames first, then height and width
    for path, subdirs, files in os.walk(path_to_vids):
        for vid in files:
            if (vid == ".DS_Store"):
                continue

            count += 1

            if (count < 725):
                continue

            print(count)

            # Get an array of the video
            vid_array = convertVidToArray(path + '/' + vid, N_frames)

            # Trim height and width of video to be of shape H x W
            vid_array = resolution(vid_array, H, W)

            # Set source and destination
            source = vid[0:len(vid)-3] + "avi"
            destination = "preprocessed_data" + path[4:]

            if ((vid_array.shape[1], vid_array.shape[2], 3) == (H, W, 3)):
                # Open video writer
                out = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H), True)

                for frame in range(N_frames):
                    out.write(vid_array[frame])

                out.release()
                move(source, destination)

def countVids(path_to_vids):
    count = 0
    for path, subdirs, files in os.walk(path_to_vids):
        for vid in files:
            count += 1

    print(count)
