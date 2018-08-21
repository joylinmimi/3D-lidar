import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from itertools import cycle
#########################
fig = plt.figure()
ax = p3.Axes3D(fig)
##############################
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
center_save_x=[]
center_save_y=[]
center_save_z=[]
###########################################
f=open('/home/joy/r-fans/video/change_axis.pcapng','rb')
data=f.read()
data = bytearray(data)
packetIndex=-1;
count=0;
a=np.arange(-1600,1000)
b=np.zeros(len(a))
b2=np.zeros(len(a))
#####################################
#simulate of tunnel
for l in range(np.size(a)):
    b[l]=-(3*a[l])-2350  #lower line
    b2[l]=-(3*a[l])+900  #upper line
#####################################
#output data per turn
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
                #print len(x_all) ,x_all.shape, type(x_all)
##########################################
                count=0
                cluster_members_x=[]
                cluster_members_y=[]
                cluster_members_z=[]
                cluster_center_x=[]
                cluster_center_y=[]
                cluster_center_z=[]
                in_range_x=[]
                in_range_y=[]
                in_range_z=[]
                not_in_range=[]
                center_color=[]
                center_size=[]
                color_array=[]
                color=[]
                text=[]
################################################################################
#data inside the tunnle
                mask=(3*x_all+ y_all>-2350) & (3*x_all+y_all<800) & (z_all>0)
                x_mask=x_all[mask]
                y_mask=y_all[mask]
                z_mask=z_all[mask]
                dstack=[[]]
                dstack=np.dstack((x_mask,y_mask,z_mask)).tolist()
                dstack=dstack[0]
                dstack = np.asarray(dstack)
##################################################################################
#mean shift
                #print dstack.shape ,type(dstack)
                bandwidth = estimate_bandwidth(dstack, quantile=0.6, n_samples=80)

                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(dstack)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                labels_unique = np.unique(labels)
                n_clusters_ = len(labels_unique)
                colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

                for k, col in zip(range(n_clusters_), colors):
                    cluster_center_x.append(0)
                    cluster_center_y.append(0)
                    cluster_center_z.append(0)
                    my_members = labels == k
                    cluster_center = cluster_centers[k]
                    for m in range(len(dstack[my_members,0])):
                        color.extend(col)
                    if len(dstack[my_members, 0]) > 740:
                        center_color.append('y')
                        center_size.append(1200)
                    else:
                        center_color.append('r')
                        center_size.append(200)
                    cluster_members_x.extend(dstack[my_members, 0])
                    cluster_members_y.extend(dstack[my_members, 1])
                    cluster_members_z.extend(dstack[my_members, 2])
#####################################################################################                    
#find stopped objects
                    for n in range(len(center_save_x)):

                        if ((cluster_centers[k][0]-center_save_x[n])**2+(cluster_centers[k][1]-
                                    center_save_y[n])**2+(cluster_centers[k][2]-center_save_z[n])**2)<24000:

                            in_range_x.append(cluster_centers[k][0])
                            in_range_y.append(cluster_centers[k][1])
                            in_range_z.append(cluster_centers[k][2])
                            not_in_range.append(n)  
                    cluster_center_x[k]=cluster_center[0]
                    cluster_center_y[k]=cluster_center[1]
                    cluster_center_z[k]=cluster_center[2]

                    color_array.extend(color)
####################################################################################################
#moving objects(delete stopped objects) 
                center_index=[]
                for o in range(len(cluster_center_x)):

                    if cluster_center_x[o] in in_range_x :
                        center_index.append(cluster_center_x.index(cluster_center_x[o]))

                cluster_center_x_not_moving=np.delete(cluster_center_x, center_index)
                cluster_center_y_not_moving=np.delete(cluster_center_y, center_index)
                cluster_center_z_not_moving=np.delete(cluster_center_z, center_index)
                center_color=np.delete(center_color,center_index)
                center_size=np.delete(center_size,center_index)
#####################################################################################################
#plot result
                ax.clear()
                ax.set_xlim3d([-1500,1500]) 
                ax.set_ylim3d([-1500,1500]) 
                ax.set_zlim3d([-500,500])
                ax.plot(a,b,0,'r')
                ax.plot(a,b2,0,'r')
                ax.scatter(cluster_members_x, cluster_members_y,cluster_members_z,s=0.5,c=color)
                ax.scatter(cluster_center_x_not_moving, cluster_center_y_not_moving,0,linewidth=4, c=center_color, alpha=0.5,s=center_size)
                center_save_x=cluster_center_x
                center_save_y=cluster_center_y
                center_save_z=cluster_center_z
                plt.pause(1e-17)        

plt.show()
f.close()
