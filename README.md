# PySLAM

A toy SLAM implmenetation in python.

Heavily inspired by geohot's [twitchslam](https://github.com/geohot/twitchslam)

## Requirements

- [g2o](https://github.com/uoip/g2opy)

## Setup

```
pip3 install -r requirements.txt
```

## Usage

```
python3 PySLAM/slam.py./slam.py data/country-road-driving.mp4
```

<img width=600px src="https://raw.githubusercontent.com/pAplakidis/PySLAM/master/images/Display2D.png" />

<img width=600px src="https://raw.githubusercontent.com/pAplakidis/PySLAM/master/images/Display3D.png" />

### Theory

- [Understanding Monocular SLAM implementation in Python OpenCV](https://learnopencv.com/monocular-slam-in-python/#aioseo-monocular-visual-slam-algorithm-outline)

### Known Issues

- 3D points are wrong when it comes to actual shapes (check freiburgdesk example)

### TODO

- Use matches to avoid duplicate 3D points
- Bundle Adjustment (DONE)
- Loop Closure
- Pose Graph Optmiziation
