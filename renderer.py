#!/usr/bin/env python3
import numpy as np
import OpenGL.GL as gl
import pangolin
from multiprocessing import Process, Queue

from utils import *

class Renderer:
  def __init__(self, w, h):
    self.W = w
    self.H = h
    self.poses = None
    self.q = Queue()
    self.p = Process(target=self.renderer_main, args=(self.q,))
    #self.p.daemon = True
    self.p.start()

  def display_init(self):
    pangolin.CreateWindowAndBind('Main', self.W, self.H)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pangolin.OpenGlRenderState(pangolin.ProjectionMatrix(self.W, self.H, 420, 420, self.W//2, self.H//2, 0.2, 100),
                                      pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(self.scam)

    self.dcam = pangolin.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    self.dcam.SetHandler(handler)

  def renderer_main(self, q):
    print("Initializing 3D Display ...")
    self.display_init()

    while not pangolin.ShouldQuit():
    #while True:
      while not q.empty():
        self.poses = q.get()

      gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
      self.dcam.Activate(self.scam)

      # TODO: poses are wrong (no translation and wrong rotation)
      # Draw camera
      if self.poses is not None:
        #print("Map pose:")
        #print(self.pose)
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCameras(self.poses, 0.5, 0.75, 0.8)

        # TODO: handle points in the map as well
        """
        points = np.random.random((100000, 3)) * 10
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(points)
        """

      pangolin.FinishFrame()

  def draw(self, frames):
    if self.q is None:
      return
    
    poses = []
    for frame in frames:
      #pose = np.identity(4)
      #pose[:3, 3] = TfromRt(frame.pose)
      #poses.append(pose)
      poses.append(frame.pose)
    self.q.put(np.array(poses))

