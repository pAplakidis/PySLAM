import open3d as o3d
import numpy as np

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
  def __init__(self):
    self.vis = o3d.visualization.Visualizer()
    self.vis.create_window(window_name="Display 3D")
    opt = self.vis.get_render_option()
    opt.background_color = np.array([0,0,0])

    self.frustums = []
    self.all_points = np.zeros((0,3))
    self.pcd = o3d.geometry.PointCloud()
    self.vis.add_geometry(self.pcd)

  def draw(self, frames):
    for f in frames[len(self.frustums):]:
      # poses
      fr = create_camera_frustum()
      fr.transform(f.pose)
      self.vis.add_geometry(fr)
      self.frustums.append(fr)

      # points
      if f.points is not None:
        if self.all_points.size == 0:
          self.all_points = f.points.copy()
        else:
          self.all_points = np.vstack([self.all_points, f.points])

        self.pcd.points = o3d.utility.Vector3dVector(self.all_points)
        self.pcd.paint_uniform_color([0.7,0.7,0.7])
        self.vis.update_geometry(self.pcd)

    self.vis.poll_events()
    self.vis.update_renderer()

  def close(self):
    self.vis.destroy_window()
