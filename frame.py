#!/usr/bin/env python3
import sys
import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import AffineTransform

from utils import *

ORB = cv2.ORB_create()
BF = cv2.BFMatcher(cv2.NORM_HAMMING)

CULLING_ERR_THRES = 0.02
RANSAC_RESIDUAL_THRES = 0.02
RANSAC_MAX_TRIALS = 100


def extract_features(frame):
  frame.img = cv2.resize(frame.img, (W, H))
  kps, des = ORB.detectAndCompute(frame.img, None)
  frame.kps = kps
  frame.des = des
  # for x in kps:
  #   print("({:.2f},{:.2f}) = size {:.2f} angle {:.2f}".format(
  #           x.pt[0], x.pt[1], x.size, x.angle))

  for kp in kps:
    x, y = int(kp.pt[0]), int(kp.pt[1])
    cv2.circle(frame.img, (x, y), radius=3, color=(0, 255, 0))

  return frame

def match_frames(f1, f2):
  matches = BF.knnMatch(f1.des, f2.des, k=2)

   # Lowe's ratio test
  good_matches = []
  for m, n in matches:
    if m.distance < 0.75 * n.distance:
      if m.distance < 32:
        good_matches.append(m)

  # Extract the matched keypoints
  src_pts = np.float32([f1.kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
  dst_pts = np.float32([f2.kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

  # FIXME: shouldn't fail
  # Apply RANSAC to filter out outliers
  model, inliers = None, None
  try:
    model, inliers = ransac((src_pts, dst_pts),
                            EssentialMatrixTransform,
                            min_samples=8,
                            residual_threshold=RANSAC_RESIDUAL_THRES,
                            max_trials=RANSAC_MAX_TRIALS)
  except Exception as e:
    pass

  if inliers is None or np.sum(inliers) < 8:
    return [], np.zeros((4, 4))
  
  inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]]

  # Draw lines between matching keypoints
  for match in inlier_matches:
    pt1 = f1.kps[match.queryIdx].pt
    pt2 = f2.kps[match.trainIdx].pt
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))

    cv2.line(f2.img, pt1, pt2, color=(255, 0, 0), thickness=2)
  
  return inlier_matches, fundamentalToRt(model.params)


class Frame:
  def __init__(self, img):
    self.img = img
    self.kps = None
    self.des = None


if __name__ == "__main__":
  cap = cv2.VideoCapture(sys.argv[1])

  frames = []

  while True:
    ret, frame = cap.read()

    if not ret:
      break

    # process frame
    frame = Frame(frame)
    frame = extract_features(frame)
    frames.append(frame)

    # match frames
    if not len(frames) > 2:
      continue
    
    # match frames
    f1, f2 = frames[-2], frames[-1]
    inlier_matches, Rt = match_frames(f1, f2)
    print(Rt)

    # display image
    cv2.imshow("frame", f2.img)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
      break

  cap.release()
  cv2.destroyAllWindows()
