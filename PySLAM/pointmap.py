import numpy as np
from typing import List

class PointMap:
  def __init__(self):
    self.poses: List[np.ndarray] = []
    self.points: np.ndarray = np.empty((0, 3))
    self.colors: np.ndarray = np.empty((0, 3))  # img pixel colors of 3D points

  def add_observation(self, points: np.ndarray, pose: np.ndarray, colors: np.ndarray):
    self.poses.append(pose)
    start_idx = len(self.points)

    if self.points.size == 0:
      self.points = points.copy()
    else:
      self.points = np.vstack([self.points, points])

    if self.colors.size == 0:
      self.colors = colors.copy()
    else:
      self.colors = np.vstack([self.colors, colors])

    return list(range(start_idx, len(self.points)))
