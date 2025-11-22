#!/usr/bin/env python3
import sys
import cv2
import time
import g2o
import numpy as np
import multiprocessing as mp
from typing import Tuple, Optional, List

from utils import *
from constants import *
from frame import Frame, match_frames
from pointmap import PointMap
from display3d import Display3D
from optimizer import BundleAdjustment, PoseGraphOptimization


class Slam:
  def __init__(self, video_path, W, H, K=None):
    self.video_path = video_path
    self.W = W
    self.H = H
    self.K = K if K is not None else np.array([[F, 0, W/2], [0, F, H/2], [0, 0, 1]])  # TODO: proper camera calibration
    self.frames: List[Frame] = []
    self.mapp = PointMap()
    self._next_point_id = 0
    self.point_id_map = {}  # maps map-index -> global BA point ID

    mp.set_start_method("spawn")  # MacOS
    self.cap = cv2.VideoCapture(sys.argv[1])
    self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    self.disp3d = Display3D(self.W, self.H, max_frames=self.n_frames)

    self.ba = BundleAdjustment()
    self.cam = g2o.CameraParameters(F, PRINCIPAL_POINT, 0)
    self.cam.set_id(0)
    self.ba.add_parameter(self.cam)

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
    valid_indices = np.where(mask)[0]
    valid_descriptors = f2.des[idx2][valid_indices]
    X = X[:,mask]
    f2.points = X.T       # N×3

    # get colors of corresponding pixels
    pixel_coords = f2.kpus[valid_indices].astype(np.int32)
    pixel_coords = pixel_coords[:, ::-1]
    colors = f2.img[pixel_coords[:, 0], pixel_coords[:, 1]]  # N×3
    colors = colors[:, ::-1] / 255.0

    # triangulated points to world coordinates
    R1 = f1.pose[:3, :3]  # f1.pose is assumed to be world_from_camera1 transform
    t1 = f1.pose[:3, 3].reshape(3, 1)
    X_world = (R1 @ X) + t1        # (3×N)
    f2.points_world = X_world.T    # (N×3)

    # return f2.points, valid_descriptors, colors
    return f2.points_world, valid_descriptors, colors

  def optim_step(self, keyframe: Frame, map_indices: List):
    self.ba.add_pose(keyframe.fid, keyframe.pose)
    kp_3D_indices = np.where(keyframe.kp_has_3D)[0]
    for i, map_idx in enumerate(map_indices):
      point_id = map_idx
      point_3D = self.mapp.points[map_idx]

      self.ba.add_point(point_id, point_3D)

      measurement = keyframe.kpus[kp_3D_indices[i]]   # pixel coordinates
      self.ba.add_edge(point_id, keyframe.fid, measurement)
    self.ba.optimize()

  def step(
      self,
      idx: int,
      img: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Frame], Optional[Frame]]:
    # process frame
    frame = Frame(idx, self.K, img)
    self.frames.append(frame)

    # match frames
    if len(self.frames) < 2: return None, None, None, None
    f1, f2 = self.frames[-2], self.frames[-1]
    idx1, idx2, Rt = match_frames(f1, f2)

    # update pose
    f2.pose = f1.pose @ Rt
    print("pose:", f2.pose)

    # update pointmap
    # Only use matches where neither keypoint already has a 3D point
    mask = ~f1.kp_has_3D[idx1] & ~f2.kp_has_3D[idx2]
    new_idx1 = idx1[mask]
    new_idx2 = idx2[mask]

    if len(new_idx1) == 0:
      return idx1, idx2, f1, f2  # nothing new to triangulate

    points, descriptors, colors = self.triangulate_points(new_idx1, new_idx2, f1, f2)

    # mark keypoints as now having 3D points
    f1.kp_has_3D[new_idx1] = True
    f2.kp_has_3D[new_idx2] = True

    map_indices = self.mapp.add_observation(points, descriptors, f2.pose, colors)
    self.optim_step(f2, map_indices)

    return idx1, idx2, f1, f2

  def display_step(self, idx1: int, idx2: int, f1: Frame, f2: Frame):
    self.disp3d.draw(self.mapp)
    f2.annotate_img(idx1, idx2, f1, f2)
    cv2.imshow("Display 2D", f2.img)

  def run(self):
    i = 0
    while True:
      start_time = time.time()
      ret, img = self.cap.read()
      if not ret:
        break
      print(f"Frame {i+1}/{self.n_frames}")

      idx1, idx2, f1, f2 = self.step(i, img)
      if idx1 is None or idx2 is None or f1 is None or f2 is None:
        continue

      self.display_step(idx1, idx2, f1, f2)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      i += 1
      print(f"[perf] Time: {(time.time()-start_time)*1000.0:.2f} ms\n")

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
