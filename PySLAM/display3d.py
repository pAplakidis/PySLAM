import numpy as np
import open3d as o3d
from multiprocessing import Process, Queue

from pointmap import PointMap

def create_camera_frustum(scale=0.2, color=[0,1,0]):
  pts = np.array([
    [0, 0, 0],
    [-0.5, -0.5, -1],
    [ 0.5, -0.5, -1],
    [ 0.5,  0.5, -1],
    [-0.5,  0.5, -1],
  ]) * scale
  lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
  fr = o3d.geometry.LineSet()
  fr.points = o3d.utility.Vector3dVector(pts)
  fr.lines  = o3d.utility.Vector2iVector(lines)
  fr.colors = o3d.utility.Vector3dVector([color]*len(lines))
  return fr


class Display3D:
  def __init__(self, W: int, H: int, max_frames=1000):
    self.W = W
    self.H = H
    self.max_frames = max_frames

    self.state = None
    self.frustums = []
    self.points = np.zeros((0,3))
    self.fid = 0
    self.q = Queue()

    self.vp = Process(target=self.viewer_thread, args=(self.q,), daemon=True)
    self.vp.start()

  def init_display(self):
    print("[Display3D] Initializing ...")

    # init visualizer
    self.vis = o3d.visualization.Visualizer()
    self.vis.create_window(window_name="Display 3D", width=self.W, height=self.H)
    opt = self.vis.get_render_option()
    opt.background_color = np.array([0,0,0])

    # zoom behavior
    ctr = self.vis.get_view_control()
    ctr.set_constant_z_far(1000.0)
    ctr.set_constant_z_near(0.01)
    ctr.set_zoom(0.5)

    # pre-render cameras and pointcloud
    for _ in range(self.max_frames):
      fr = create_camera_frustum()
      self.vis.add_geometry(fr)
      self.frustums.append(fr)

    self.pcd = o3d.geometry.PointCloud()
    self.vis.add_geometry(self.pcd)

    print("[Display3D] Init done")

  def viewer_thread(self, q: Queue):
    self.init_display()
    while True:
      self.tick(q)

  def tick(self, q: Queue):
    while not q.empty():
      self.state = q.get()

    if self.state is None:
      self.vis.poll_events()
      self.vis.update_renderer()
      return

    poses, points = self.state
    print(f"[renderer] poses: ({len(poses)}x{poses[0].shape}) - points: {points.shape}")

    # poses
    curr_fid = self.fid
    for k, pose in enumerate(poses[curr_fid:]):
      j = curr_fid + k
      self.frustums[j].transform(pose)           
      self.vis.update_geometry(self.frustums[j])
      self.fid += 1

    # TODO: color points based on pixel color
    # points
    self.points = points
    self.pcd.points = o3d.utility.Vector3dVector(self.points)
    self.pcd.paint_uniform_color([0.7,0.7,0.7])
    self.vis.update_geometry(self.pcd)

    self.vis.poll_events()
    self.vis.update_renderer()

  def draw(self, mapp: PointMap):
    if self.q is None:
      return

    poses, points = mapp.poses.copy(), np.copy(mapp.points)
    self.q.put((poses, np.vstack(points)))

  def close(self):
    self.vis.destroy_window()
