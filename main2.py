import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import copy


#Code for Processing
class PointCloudProcessor:
    def __init__(self):
        self.pcd = None
        self.segments = {}
        self.bounding_boxes = {}

    def load_point_cloud(self, filename):
        """Load point cloud from file."""
        self.pcd = o3d.io.read_point_cloud(filename)
        return self.pcd

    def preprocess(self, voxel_size=0.02):
        """Preprocess point cloud with downsampling and normal estimation."""
        #Downsampling
        self.pcd = self.pcd.voxel_down_sample(voxel_size)

        # Estimate normals
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        return self.pcd

    def segment_plane(self, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        """Segment planes using RANSAC."""
        plane_segments = []
        remaining_cloud = copy.deepcopy(self.pcd)

        while len(remaining_cloud.points) > 100:  #Continue until few points remain
            # Segment plane
            plane_model, inliers = remaining_cloud.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )

            # Extract plane and non-plane points
            plane_cloud = remaining_cloud.select_by_index(inliers)
            remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)

            # Store plane segment
            if len(plane_cloud.points) > 100:  # Only keep significant planes
                plane_segments.append(plane_cloud)

        self.segments['planes'] = plane_segments
        return plane_segments

    def cluster_objects(self, eps=0.05, min_points=10):
        """Cluster remaining points using DBSCAN."""
        # Convert point cloud to numpy array
        points = np.asarray(self.pcd.points)

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
        labels = clustering.labels_

        # Separate clusters
        clusters = []
        for label in set(labels):
            if label != -1:  # Ignore noise points
                cluster_points = points[labels == label]
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                clusters.append(cluster_pcd)

        self.segments['clusters'] = clusters
        return clusters #return

    def compute_bounding_boxes(self):
        """Compute bounding boxes for all segments."""
        for segment_type, segments in self.segments.items():
            self.bounding_boxes[segment_type] = []
            for segment in segments:
                bbox = segment.get_axis_aligned_bounding_box()
                bbox.color = (1, 0, 0)  # Red bounding box
                self.bounding_boxes[segment_type].append(bbox)

    def get_object_dimensions(self, segment):
        """Get dimensions of a segment."""
        bbox = segment.get_axis_aligned_bounding_box()
        return bbox.get_extent()

    def visualize(self, show_normals=False):
        """Visualize processed point cloud with segments and bounding boxes."""
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add original point cloud
        vis.add_geometry(self.pcd)

        # Add segments with different colors
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
        for segment_type, segments in self.segments.items():
            for i, segment in enumerate(segments):
                segment.paint_uniform_color(colors[i % len(colors)])
                vis.add_geometry(segment)

        # Add bounding boxes
        for bbox_type, bboxes in self.bounding_boxes.items():
            for bbox in bboxes:
                vis.add_geometry(bbox)

        if show_normals:
            vis.add_geometry(self.pcd.compute_point_cloud_normal_visualization())

        vis.run()
        vis.destroy_window()

    def export_processed_cloud(self, filename):
        """Export processed point cloud to file."""
        o3d.io.write_point_cloud(filename, self.pcd)

def main():
    # Initialize processor
    processor = PointCloudProcessor()

    # Load point cloud
    processor.load_point_cloud(o3d.data.PLYPointCloud().path)

    # Preprocess
    processor.preprocess(voxel_size=0.02)
    print("done processing 1\n")

    # Segment planes
    processor.segment_plane(distance_threshold=0.01)
    print("done processing 2\n")

    # Cluster remaining objects
    # print(np.asarray(processor.pcd.points))
    processor.cluster_objects(eps=0.05, min_points=10)
    print("done processing 3\n")

    # Compute bounding boxes
    processor.compute_bounding_boxes()
    print("done processing 4\n")

    # Visualize results
    processor.visualize(show_normals=False)

    # Export processed cloud
    processor.export_processed_cloud("processed_cloud.pcd")

if __name__ == "__main__":
    main()
