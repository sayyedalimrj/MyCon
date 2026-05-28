import sys
import cv2
import numpy as np
import open3d as o3d
import ifcopenshell

print("Python:", sys.version)
print("OpenCV:", cv2.__version__)
print("Open3D:", o3d.__version__)
print("IfcOpenShell:", ifcopenshell.version)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
o3d.io.write_point_cloud("/tmp/smoke.ply", pcd)

print("CORE_SMOKE_OK")