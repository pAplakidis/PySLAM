#!/usr/bin/env python3
import numpy as np
import open3d as o3d

def create_camera_frustum(scale=0.2, color=[0, 1, 0]):
  """
  Create a simple SLAM-style camera frustum (square + tail).
  Returns an Open3D LineSet.
  """
  # Frustum vertices (unit square at Z=1)
  pts = np.array([
    [0, 0, 0],        # camera center
    [-0.5, -0.5, 1],  # bottom-left
    [ 0.5, -0.5, 1],  # bottom-right
    [ 0.5,  0.5, 1],  # top-right
    [-0.5,  0.5, 1],  # top-left
  ]) * scale

  # Edges: center to corners, and square edges
  lines = [
    [0, 1], [0, 2], [0, 3], [0, 4],  # tail connections
    [1, 2], [2, 3], [3, 4], [4, 1]   # square edges
  ]

  frustum = o3d.geometry.LineSet()
  frustum.points = o3d.utility.Vector3dVector(pts)
  frustum.lines = o3d.utility.Vector2iVector(lines)
  frustum.colors = o3d.utility.Vector3dVector([color] * len(lines))
  return frustum


def draw_scene(pointcloud_np, poses):
  """
  Render a SLAM scene with point cloud and camera poses.
  pointcloud_np : (N, 3) numpy array of points
  poses         : list of 4x4 numpy arrays (camera extrinsics)
  """
  # Create point cloud object
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pointcloud_np)
  pcd.paint_uniform_color([0.7, 0.7, 0.7])

  # Create camera frustums for each pose
  frustums = []
  for T in poses:
    frustum = create_camera_frustum(scale=0.3, color=[0, 1, 0])
    frustum.transform(T)
    frustums.append(frustum)

  # Draw everything
  o3d.visualization.draw_geometries([pcd] + frustums)


if __name__ == "__main__":
    # pointcloud (static)
    points = np.random.rand(2000,3)*5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.7,0.7,0.7])

    # fake poses over time
    poses_list = []
    for i in range(20):
      T = np.eye(4)
      T[:3,3] = [i*0.2, 0, 0]
      poses_list.append(T)

    # init viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window("SLAM viz")
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    vis.add_geometry(pcd)

    # list to accumulate rendered frustums
    frustums = []

    for k, T in enumerate(poses_list):
      # create ONE new frustum for this frame
      fr = create_camera_frustum(scale=0.3, color=[0,1,0])
      fr.transform(T)
      vis.add_geometry(fr)
      frustums.append(fr)

      # update viewer
      vis.poll_events()
      vis.update_renderer()

    print("Finished, press Q on the window to exit.")
    vis.run()
    vis.destroy_window()
