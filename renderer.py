import numpy as np
import open3d as o3d
from multiprocessing import Process, Queue

def create_camera_frustum(scale=0.2, color=[0,1,0]):
  pts = np.array([
    [0,0,0],
    [-0.5,-0.5,1],
    [ 0.5,-0.5,1],
    [ 0.5, 0.5,1],
    [-0.5, 0.5,1],
  ]) * scale
  lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]

  fr = o3d.geometry.LineSet()
  fr.points = o3d.utility.Vector3dVector(pts)
  fr.lines  = o3d.utility.Vector2iVector(lines)
  fr.colors = o3d.utility.Vector3dVector([color]*len(lines))
  return fr

class Renderer:
  def __init__(self, w, h):
    self.W = w
    self.H = h
    self.q = Queue()
    self.p = Process(target=self.renderer_main, args=(self.q,))
    self.p.start()

  def renderer_main(self, q: Queue):
    vis = o3d.visualization.Visualizer()
    vis.create_window("Display 3D")#, width=self.W, height=self.H)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])

    # static scene objects you might add later: map points, axes, etc.
    geometries = []   # list of LineSets for cameras
    latest_poses = None

    while True:
      # pull latest poses if any
      while not q.empty():
        latest_poses = q.get()

      if latest_poses is not None:
        # clear old cameras from viewer
        for g in geometries:
          vis.remove_geometry(g, reset_bounding_box=False)
        geometries.clear()

        # add new frustums
        for T in latest_poses:
          fr = create_camera_frustum(scale=0.3)
          fr.transform(T)
          vis.add_geometry(fr, reset_bounding_box=False)
          geometries.append(fr)

      vis.poll_events()
      vis.update_renderer()

  def draw(self, frames):
    poses = [f.pose for f in frames]
    self.q.put(np.array(poses))
