#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import multiprocessing as mp

from utils import *
from frame import Frame, match_frames
from renderer import Renderer

class Slam:
  def __init__(self):
    pass


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python3 frame.py <video_file>")
    sys.exit(1)

  mp.set_start_method("spawn")
  renderer = Renderer()

  # TODO: proper camera calibration
  F = 525.0
  K = np.array([[F, 0, W/2],
                [0, F, H/2],
                [0, 0, 1]])
  
  cap = cv2.VideoCapture(sys.argv[1])
  frames = []
  i = 0
  while True:
    ret, img = cap.read()
    print(f"Frame {i+1}")

    if not ret:
      break

    # process frame
    frame = Frame(i, K, img)
    frames.append(frame)

    # match frames
    if len(frames) < 2:
      continue
    
    # match frames
    f1, f2 = frames[-2], frames[-1]
    idx1, idx2, Rt = match_frames(f1, f2)
    f2.pose = np.dot(f1.pose, Rt)
    renderer.draw(frames)
    print("Rt:", Rt)

    # display image
    f2.draw_img(idx1, idx2, f1, f2)
    # renderer.vis.run()
    cv2.imshow("Display 2D", f2.img)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
      break
    i += 1
    print()

  cap.release()
  cv2.destroyAllWindows()

  print("Finished, press Q on the window to exit.")
  renderer.close()
