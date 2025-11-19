# optimizer.py
import g2o
import numpy as np


# Offset to avoid id collisions between pose and point vertices.
# You can raise this if you have many poses/points.
POINT_ID_OFFSET = 1000000


class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, cam_parameter_id=0):
        """
        cam_parameter_id: the id of the CameraParameters you add to the optimizer (slam.py sets it to 0)
        """
        super().__init__()

        # Linear solver + block solver for SE3
        linear_solver = g2o.LinearSolverEigenSE3()
        block_solver = g2o.BlockSolverSE3(linear_solver)
        algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)
        self.set_algorithm(algorithm)

        self.cam_param_id = cam_parameter_id

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)

    # ---------- Pose (SE3) ----------
    def add_pose(self, pose_id: int, pose: np.ndarray, fixed: bool = False):
        """
        pose: 4x4 numpy array (SE3)
        pose_id: integer id (will be used directly as vertex id)
        """
        R = pose[:3, :3]
        t = pose[:3, 3]
        se3 = g2o.SE3Quat(R, t)

        v = g2o.VertexSE3Expmap()
        v.set_id(pose_id)
        v.set_estimate(se3)
        v.set_fixed(fixed)

        self.add_vertex(v)

    # ---------- Point (XYZ) ----------
    def add_point(self, point_id: int, point: np.ndarray, fixed: bool = False):
        """
        point: (3,) numpy array
        point_id: integer id (will be offset to avoid colliding with pose ids)
        """
        v = g2o.VertexPointXYZ()
        v.set_id(POINT_ID_OFFSET + point_id)
        v.set_estimate(point)
        v.set_fixed(fixed)
        # Some builds support set_marginalized; if present, set it:
        try:
            v.set_marginalized(True)
        except Exception:
            pass

        self.add_vertex(v)

    # ---------- Reprojection edge: projects 3D point to 2D measurement ----------
    def add_edge(self, point_id: int, pose_id: int,
                 measurement: np.ndarray,
                 information: np.ndarray = None,
                 robust_kernel: g2o.RobustKernelHuber = None):
        """
        measurement: 2-vector (u, v) image coordinates (same coords your slam.py passes)
        point_id: integer id used for add_point (we will offset internally)
        pose_id: integer id used for add_pose
        information: 2x2 information matrix (optional)
        robust_kernel: optional robust kernel (pass e.g. g2o.RobustKernelHuber(...))
        """
        if information is None:
            information = np.eye(2)

        e = g2o.EdgeProjectXYZ2UV()

        # ordering: vertex 0 = point, vertex 1 = pose (SE3Expmap)
        e.set_vertex(0, self.vertex(POINT_ID_OFFSET + point_id))
        e.set_vertex(1, self.vertex(pose_id))

        e.set_measurement(measurement)
        e.set_information(information)

        # Attach camera parameters (assumes you added CameraParameters with id self.cam_param_id)
        # Some wheels require parameter index 0; others accept the actual id. We'll set the id that
        # matches how you added the parameter in slam.py (cam.set_id(0)).
        try:
            e.set_parameter_id(0, self.cam_param_id)
        except Exception:
            # Some builds expose a different API; ignore if not available.
            pass

        if robust_kernel is not None:
            e.set_robust_kernel(robust_kernel)

        # self.add_edge(e)
        if robust_kernel is not None:
            e.set_robust_kernel(robust_kernel)
        super().add_edge(e)


    # ---------- Getters ----------
    def get_pose(self, pose_id: int) -> np.ndarray:
        v = self.vertex(pose_id)
        est = v.estimate()  # SE3Quat
        R = est.rotation().matrix()
        t = est.translation()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_point(self, point_id: int) -> np.ndarray:
        return self.vertex(POINT_ID_OFFSET + point_id).estimate()


# -------------------------------
# Pose Graph (keeps mostly same)
# -------------------------------
class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        linear_solver = g2o.LinearSolverEigenSE3()
        block_solver = g2o.BlockSolverSE3(linear_solver)
        solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
        self.set_algorithm(solver)

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id: int, pose: np.ndarray, fixed: bool = False):
        R = pose[:3, :3]
        t = pose[:3, 3]
        se3 = g2o.SE3Quat(R, t)

        v = g2o.VertexSE3Expmap()
        v.set_id(id)
        v.set_estimate(se3)
        v.set_fixed(fixed)
        super().add_vertex(v)

    def add_edge(self, vertices, measurement: np.ndarray, information=np.identity(6), robust_kernel=None):
        e = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            e.set_vertex(i, v)

        R = measurement[:3, :3]
        t = measurement[:3, 3]
        e.set_measurement(g2o.SE3Quat(R, t))
        e.set_information(information)
        if robust_kernel is not None:
            e.set_robust_kernel(robust_kernel)
        super().add_edge(e)

    def get_pose(self, id: int) -> np.ndarray:
        est = self.vertex(id).estimate()
        R = est.rotation().matrix()
        t = est.translation()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
