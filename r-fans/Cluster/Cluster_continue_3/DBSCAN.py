from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN
from extra_functions import cluster_gen
import pcl
import numpy as np
import matplotlib.cm as cm
def DBScan():

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
	#print data
	#file=open("data.txt","w")
	#file.write(data)
	#print type(data)
	# Define max_distance (eps parameter in DBSCAN())
	max_distance = 175 
	#max_distance = 1000
	db = DBSCAN(eps=max_distance, min_samples=10).fit(data)
	# Extract a mask of core cluster members
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	# Extract labels (-1 is used for outliers)
	labels = db.labels_
	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	unique_labels = set(labels)
	#print unique_labels
	# Plot up the results!
	# The following is just a fancy way of plotting core, edge and outliers
	colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        array_x=[]
        array_y=[]
        array_z=[]

	for k, col in zip(unique_labels, colors):
	    #ax.clear()
            if k == -1:
		# Black used for noise.
		#col = [0, 0, 0, 1]
		col =(0,0,0,1)
	
	    class_member_mask = (labels == k)
	    #print len(class_member_mask)
	    xy2 = data[class_member_mask & core_samples_mask]
            array_x.extend(xy2[:,0])
            array_y.extend(xy2[:,1])
            array_z.extend(xy2[:,2])
	stack_array=np.vstack((array_x,array_y,array_z))
            #print xy2
	return stack_array
	    #print test
	    #print tuple(col)

	    #ax.scatter(xy[:, 0], xy[:, 1], xy[:,2],color=col)
	    #ax.clear()
	#plt.show()

