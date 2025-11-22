import numpy as np
from typing import List

class PointMap:
  def __init__(self):
    self.poses: List[np.ndarray] = []
    self.descriptors = []
    self.points: np.ndarray = np.empty((0, 3))
    self.colors: np.ndarray = np.empty((0, 3))  # img pixel colors of 3D points

  def add_observation(self, points: np.ndarray, descriptors: np.ndarray, pose: np.ndarray, colors: np.ndarray):
    self.poses.append(pose)

    start_idx = len(self.points)
    if len(points) > 0 and len(descriptors) > 0:
      valid_points = []
      unique_descriptors = set()
      for i, (p, d) in enumerate(zip(points, descriptors)):
        descriptor_tuple = tuple(d)  # convert descriptor to a tuple for set operations
        if descriptor_tuple not in unique_descriptors:
          unique_descriptors.add(descriptor_tuple)
          valid_points.append(p)
      valid_points = np.array(valid_points)
      print(f"[pointmap] Adding {len(valid_points)}/{len(points)} points to map")

      if self.points.size == 0:
        self.points = valid_points.copy()
      else:
        self.points = np.vstack([self.points, valid_points])

    if len(colors) > 0:
      if self.colors.size == 0:
        self.colors = colors.copy()
      else:
        self.colors = np.vstack([self.colors, colors])

    return list(range(start_idx, len(self.points)))
