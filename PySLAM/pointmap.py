import numpy as np
from typing import List

class PointMap:
  def __init__(self):
    self.poses: List[np.ndarray] = []
    self.points: np.ndarray = np.empty((0, 3))

  def add_observation(self, points: np.ndarray, pose: np.ndarray):
    self.poses.append(pose)

    print(self.points.shape)
    print(points.shape)

    if self.points.size == 0:
      self.points = points.copy()
    else:
      self.points = np.vstack([self.points, points])
