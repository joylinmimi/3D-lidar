#!/usr/bin/python
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN
from extra_functions import cluster_gen
import pcl
import numpy as np
import matplotlib.cm as cm
import data_image_2
import itertools
from sklearn.cluster import MeanShift, estimate_bandwidth

# get pcd file
data_image_2.plot()
# Load Point Cloud file
cloud = pcl.load_XYZRGB('./test_rgb.pcd')

# Voxel Grid Downsampling filter
################################
# Create a VoxelGrid filter object for our input point cloud
vox = cloud.make_voxel_grid_filter()

# Choose a voxel (also known as leaf) size
# Note: this (1) means 1mx1mx1m is a poor choice of leaf size   
# Experiment and find the appropriate size!
#LEAF_SIZE = 0.01   
LEAF_SIZE =45

# Set the voxel (or leaf) size  
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

# Call the filter function to obtain the resultant downsampled point cloud
cloud_filtered = vox.filter()
filename = './pcd_out/voxel_downsampled.pcd'
pcl.save(cloud_filtered, filename)

# PassThrough filter
################################
# Create a PassThrough filter object.
passthrough = cloud_filtered.make_passthrough_filter()

# Assign axis and range to the passthrough filter object.
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0
axis_max = 100
passthrough.set_filter_limits(axis_min, axis_max)

# Finally use the filter function to obtain the resultant point cloud. 
cloud_filtered = passthrough.filter()
filename = './pcd_out/pass_through_filtered.pcd'
pcl.save(cloud_filtered, filename)
# RANSAC plane segmentation
################################
# Create the segmentation object
seg = cloud_filtered.make_segmenter()

# Set the model you wish to fit 
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

# Max distance for a point to be considered fitting the model
# Experiment with different values for max_distance 
# for segmenting the table
max_distance = 0.01
seg.set_distance_threshold(max_distance)

# Call the segment function to obtain set of inlier indices and model coefficients
inliers, coefficients = seg.segment()

# Extract outliers
# Save pcd for tabletop objects
################################
extracted_outliers = cloud_filtered.extract(inliers, negative=True)
e=np.asarray(extracted_outliers)
#print e[:,:-1]
filename = './pcd_out/extracted_outliers.pcd'
pcl.save(extracted_outliers, filename)

# Generate some clusters!
data = e[:,:-1]
print data
bandwidth = estimate_bandwidth(data, quantile=0.05, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

fig = plt.figure()
ax =Axes3D(fig)
colors = itertools.cycle(["r", "b", "g","c","y","m"])
#colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    #print my_members
    cluster_center = cluster_centers[k]
    #ax.scatter(X[my_members, 0], X[my_members, 1], col + '.')
    #ax.scatter(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             #markeredgecolor='k', markersize=14)
    ax.scatter(data[my_members, 0], data[my_members, 1],color=col,s=1)
    ax.scatter(cluster_center[0], cluster_center[1],color='black',s=10)
    print len(cluster_center[0])

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
