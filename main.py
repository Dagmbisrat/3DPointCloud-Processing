import open3d as o3d

#Used for dispalying multiple procsessed / not processed 3dClouds
pcd = o3d.io.read_point_cloud('./DataSets/processed_cloud.ply')

o3d.visualization.draw_geometries([pcd])
