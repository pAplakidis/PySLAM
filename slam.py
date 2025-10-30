#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import multiprocessing as mp
from typing import Tuple, Optional, List

from utils import *
from constants import *
from frame import Frame, match_frames
from pointmap import PointMap
from display3d import Display3D

class Slam:
  def __init__(self, video_path, W, H, K=None):
    self.video_path = video_path
    self.W = W
    self.H = H
    self.K = K if K is not None else np.array([[F, 0, W/2], [0, F, H/2], [0, 0, 1]])  # TODO: proper camera calibration
    self.frames: List[Frame] = []
    self.mapp = PointMap()

    mp.set_start_method("spawn")  # MacOS
    self.cap = cv2.VideoCapture(sys.argv[1])
    self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    self.disp3d = Display3D(self.W, self.H, max_frames=self.n_frames)

  @staticmethod
  def triangulate_points(idx1: np.ndarray, idx2: np.ndarray, f1: Frame, f2: Frame) -> np.ndarray:
    T_rel = np.linalg.inv(f1.pose) @ f2.pose
    R = T_rel[:3,:3]
    t = T_rel[:3,3]

    x1, x2 = f1.kps[idx1], f2.kps[idx2]
    x1_h = np.vstack([x1.T, np.ones(x1.shape[0])])
    x2_h = np.vstack([x2.T, np.ones(x2.shape[0])])

    P1 = np.hstack([np.eye(3), np.zeros((3,1))])     # [I|0]
    P2 = np.hstack([R, t.reshape(3,1)])              # [R|t]
    X_h = cv2.triangulatePoints(P1, P2, x1_h[:2], x2_h[:2])  # gives 4×N

    X = X_h[:3] / X_h[3]  # dehomogenize => shape 3×N
    mask = X_h[3] > 0     # keep only valid (in front of camera)
    X = X[:,mask]
    f2.points = X.T       # N×3
    return X.T

  def step(
      self,
      idx: int,
      img: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Frame], Optional[Frame]]:
    # process frame
    frame = Frame(idx, self.K, img)
    self.frames.append(frame)

    # match frames
    if len(self.frames) < 2:
      return None, None, None, None
    
    # match frames
    f1, f2 = self.frames[-2], self.frames[-1]
    idx1, idx2, Rt = match_frames(f1, f2)

    # update pointmap
    f2.pose = f1.pose @ Rt
    print("pose:", f2.pose)
    points = self.triangulate_points(idx1, idx2, f1, f2)
    self.mapp.add_observation(points, f2.pose)

    return idx1, idx2, f1, f2

  def display_step(self, idx1: int, idx2: int, f1: Frame, f2: Frame):
    self.disp3d.draw(self.mapp)
    f2.annotate_img(idx1, idx2, f1, f2)
    cv2.imshow("Display 2D", f2.img)

  def run(self):
    i = 0
    while True:
      ret, img = self.cap.read()
      if not ret:
        break
      print(f"Frame {i+1}")

      idx1, idx2, f1, f2 = self.step(i, img)
      if idx1 is None or idx2 is None or f1 is None or f2 is None:
        continue

      self.display_step(idx1, idx2, f1, f2)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      i += 1
      print()

    self.cap.release()
    cv2.destroyAllWindows()
    print("Finished, press Q on the window to exit.")
    self.disp3d.close()


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python3 frame.py <video_file>")
    sys.exit(1)

  slam = Slam(sys.argv[1], W, H)
  slam.run()
