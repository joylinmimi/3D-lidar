import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import np2pcd
from sklearn.cluster import DBSCAN
import DBScan
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cv2
from extra_functions import cluster_gen
import pcl
import numpy as np
import matplotlib.cm as cm

#########################

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d([-3000,3000])
ax.set_ylim3d([-3000,3000])
ax.set_zlim3d([-500,500])

##############################

#############################################
dataID=0x96
chksum=0;
angle=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
theta=[-15,-13,-11,-9,-7,-5,-4,-3,-2,-1,0,1,3,5,7,9]
theta_degree=np.multiply(theta,np.pi/180) 
rangeOne=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
rangeTwo=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
x_all=np.zeros(3600*16)
y_all=np.zeros(3600*16)
z_all=np.zeros(3600*16)
###########################################
f=open('underbridge1_cut.pcapng','rb')
data=f.read()
data = bytearray(data)
packetIndex=-1;
count=0;
for i in range(np.size(data)):
    if i<packetIndex:
	continue
    if(data[i]==dataID):
	chksum=0
	for j in range(i+2, i+2+136):
	    chksum=chksum+data[j]
	if ((chksum&0xFF)==data[i+1]):                
	    for k in range(0,16):
		angle[k]=((data[i+11+8*k]<<8)+data[i+10+8*k])*0.01
		rangeOne[k]=((data[i+13+8*k]<<8)+data[i+12+8*k])
		rangeTwo[k]=((data[i+15+8*k]<<8)+data[i+14+8*k])
	    angle_degree=np.multiply(angle,np.pi/180)
	    xy=np.multiply(rangeOne,np.cos(theta_degree))
	    x=np.multiply(xy,np.cos(angle_degree))
	    y=-np.multiply(xy,np.sin(angle_degree))
	    z=np.multiply(rangeOne,np.sin(theta_degree))
	    if(count<3600):
		x_all[count*16:(count+1)*16]=x
		y_all[count*16:(count+1)*16]=y
		z_all[count*16:(count+1)*16]=z
	    count=count+1
	    packetIndex=i+138
	    if(count==3600):
		count=0
		#ax.clear() 
		plt.pause(1e-17)
		filename_rgb="test_rgb.pcd"
		np2pcd.np2pcd(x_all, y_all, z_all, filename_rgb, rgb=True)
		ax.clear()
		data=DBScan.DBScan()
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

		for k, col in zip(unique_labels, colors):
		    #ax.clear()
		    if k == -1:
			# Black used for noise.
			#col = [0, 0, 0, 1]
			col =(0,0,0,1)

		    class_member_mask = (labels == k)
		    xy2 = data[class_member_mask & core_samples_mask]
		    #print xy2
		    #return xy2
		    #print test
		    #print tuple(col)

		    #ax.scatter(xy[:, 0], xy[:, 1], xy[:,2],color=col)
		    #ax.clear()
		#plt.show()

		print xy2
		
		#ax.clear()
		ax.set_xlim3d([-3000,3000]) 
		ax.set_ylim3d([-3000,3000]) 
		ax.set_zlim3d([-500,500])
		ax.scatter(xy2[:, 0], xy2[:, 1], xy2[:,2])
		#ax.plot(xy2[:, 0], xy2[:, 1], xy2[:,2],'g.',markersize=1)
		plt.pause(1e-17)

		
plt.show()
f.close()

