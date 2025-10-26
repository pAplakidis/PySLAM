#!/usr/bin/env python3
import cv2
import numpy as np
from skimage.measure import ransac
from scipy.spatial import cKDTree

from utils import *
from constants import *


def match_frames(f1, f2):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(f1.des, f2.des, k=2)

  # Lowe's ratio test
  ret = []
  idx1, idx2 = [], []
  idx1s, idx2s = set(), set()
  for m, n in matches:
    if m.distance < 0.75 * n.distance:
      # query is src (f1), train is dest (f2)
      p1 = f1.kps[m.queryIdx]
      p2 = f2.kps[m.trainIdx]

      # orb distance 32
      if m.distance < 32:
        if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
          idx1.append(m.queryIdx)
          idx2.append(m.trainIdx)
          idx1s.add(m.queryIdx)
          idx2s.add(m.trainIdx)
          ret.append((p1, p2))

  # no duplicates
  assert len(set(idx1)) == len(idx1)
  assert len(set(idx2)) == len(idx2)
  assert len(ret) >= 8
  ret = np.array(ret)
  idx1 = np.array(idx1)
  idx2 = np.array(idx2)

  # Apply RANSAC to filter out outliers
  model, inliers = ransac(
    (ret[:, 0], ret[:, 1]),
    EssentialMatrixTransform,
    min_samples=8,
    residual_threshold=RANSAC_RESIDUAL_THRES,
    max_trials=RANSAC_MAX_TRIALS
  )
  print("Matches:  %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))
  return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)


class Frame:
  def __init__(self, fid, K, img):
    self.fid = 0
    self.K = K
    self.kpus = None
    self.des = None

    if img is not None:
      self.img = cv2.resize(img, (W, H))
      self.extract_features()

  @property
  def Kinv(self):
    if not hasattr(self, '_Kinv'):
      self._Kinv = np.linalg.inv(self.K)
    return self._Kinv

  # normalized keypoints
  @property
  def kps(self):
    self._kps = normalize(self.Kinv, self.kpus)
    return self._kps

  # KD tree of unnormalized keypoints
  @property
  def kd(self):
    if not hasattr(self, '_kd'):
      self._kd = cKDTree(self.kpus)
    return self._kd

  def extract_features(self):
    orb = cv2.ORB_create()
    pts = cv2.goodFeaturesToTrack(np.mean(self.img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    kps, self.des = orb.compute(self.img, kps)

    self.kpus = np.array([(kp.pt[0], kp.pt[1]) for kp in kps])
    return self.kpus, self.des

  def draw_img(self, idx1, idx2, f1, f2):
    for i, kp in enumerate(self.kpus):
      cv2.circle(self.img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 255, 0))

    for i in range(len(idx1)):
      pt1, pt2 = f1.kpus[idx1[i]].astype(np.int32), f2.kpus[idx2[i]].astype(np.int32)
      cv2.line(f2.img, pt1, pt2, color=(255, 0, 0), thickness=2)
