import cv2
import os

# This is v ugly because I was trying to get around the difficulties of testing
#  given a clean directory of just folder filled with files, would just need to
#  remove a few of the catches, and this will run
fourcc = cv2.VideoWriter_fourcc(*'XVID')
pwd = os.getcwd()

for folder in os.listdir(os.getcwd()):
  if(folder != "test2"):
    continue
  for file in os.listdir(pwd + '/' + folder):
    if(file[0] == "."):
      continue
    cur = pwd + '/' + folder + '/'
    cap = cv2.VideoCapture(cur + file)
    out = cv2.VideoWriter(cur + file[0:-4] + '_new.mp4', fourcc, 10, (320, 656) )
    success, frame = cap.read()
    count = 0

    while success:
      success, frame = cap.read()
      if not success:
        break

      if(count % 2 == 0):
        out.write(frame)

      count += 1

    # Release everything if job is finished
    cap.release()
    out.release()
    break
    # os.remove(cur)



