import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
#########################
fig = plt.figure()
ax = p3.Axes3D(fig)
'''
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d([-4000,100])
ax.set_ylim3d([-15000,1500])
ax.set_zlim3d([-500,500])
'''
#plt.show()
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

f=open('change_axis.pcapng','rb')
data=f.read()
data = bytearray(data)
packetIndex=-1;
count=0;
a=np.arange(-1600,1000)
b=np.zeros(len(a))
b2=np.zeros(len(a))
for l in range(np.size(a)):
    b[l]=-(3*a[l])-2350  #lower line
    b2[l]=-(3*a[l])+900  #higher 
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
                mask=(3*x_all+ y_all>-2350) & (3*x_all+y_all<800) & (z_all>0)
                x_mask=x_all[mask]
                y_mask=y_all[mask]
                z_mask=z_all[mask]
                dstack=[[]]
                dstack=np.dstack((x_mask,y_mask,z_mask)).tolist()
                dstack=dstack[0]
                #print type (dstack) 
                centers=dstack
                #print type (centers)
                X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

                # #############################################################################
                # Compute clustering with MeanShift

                # The following bandwidth can be automatically detected using
                bandwidth = estimate_bandwidth(X, quantile=0.5, n_samples=50)

                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(X)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                #print cluster_centers
                #print type(cluster_centers)
                labels_unique = np.unique(labels)
                n_clusters_ = len(labels_unique)
                #print cluster_centers[0]
                #print type(cluster_centers[0])
                #print cluster_centers[0].size

                #print("number of estimated clusters : %d" % n_clusters_)

                # #############################################################################
                # Plot result
                import matplotlib.pyplot as plt
                from itertools import cycle
                colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
                cluster_members_x=[]
                cluster_members_y=[]
                cluster_members_z=[]
                cluster_center_x=[]
                cluster_center_y=[]
                cluster_center_z=[]
                color_array=[]
                color=[]
                for k, col in zip(range(n_clusters_), colors):
                    #cluster_center_x=[]
                    cluster_center_x.append(0)
                    cluster_center_y.append(0)
                    cluster_center_z.append(0)
                    #print k,(cluster_center_x)
                    my_members = labels == k
                    cluster_center = cluster_centers[k]
                    #ax.clear()
                    #print x
                    #ax.scatter(X[my_members, 0], X[my_members, 1],X[my_members, 2],s=0.5,c=col)
                    for m in range(len(X[my_members,0])):
                        color.extend(col)
                    #print len(X[my_members, 0])
                    
                    #print type(X[my_members, 0])
                    #ax.scatter(cluster_center[0], cluster_center[1], cluster_center[2],s=10,c='black')
                    #print cluster_center[0].size
                
                    #plt.pause(1e-17)
                    cluster_members_x.extend(X[my_members, 0])
                    cluster_members_y.extend(X[my_members, 1])
                    cluster_members_z.extend(X[my_members, 2])
                    #print cluster_center_x[k]#,(cluster_center[0])
                    cluster_center_x[k]=cluster_center[0]
                    cluster_center_y[k]=cluster_center[1]
                    cluster_center_z[k]=cluster_center[2]
                   
                    '''
                    center_x=cluster_center[0].astype(type('float', (float,), {}))
                    center_y=cluster_center[1].astype(type('float', (float,), {}))
                    center_z=cluster_center[2].astype(type('float', (float,), {}))
                    cluster_center_x.extend(center_x)               
                    cluster_center_y.extend(center_y)
                    cluster_center_z.extend(center_z)
                    '''
                    color_array.extend(color)
           
                #print len(cluster_members_x), len(cluster_center_x)
                ax.clear()
                ax.set_xlim3d([-1500,1500]) 
                ax.set_ylim3d([-1500,1500]) 
                ax.set_zlim3d([-500,500])
                ax.plot(a,b,0,'r')
                ax.plot(a,b2,0,'r')
               
                ax.scatter(cluster_members_x, cluster_members_y,cluster_members_z,s=0.5,c=color)
                #print cluster_center_x
                #print len(cluster_center_x)
                #print cluster_centers[:,0]
                ax.scatter(cluster_center_x, cluster_center_y, cluster_center_z,s=10,c='black')
                plt.pause(1e-17)        
                
   
                '''
                ax.clear() 
                ax.set_xlim3d([-1500,1500]) 
                ax.set_ylim3d([-1500,1500]) 
                ax.set_zlim3d([-500,500])
                #ax.scatter(x_all,y_all,z_all,s=2,c='r')
                ax.plot(a,b,0,'r')
                ax.plot(a,b2,0,'r')
                ax.plot(x_mask,y_mask,z_mask,'g.',markersize=0.5)
                plt.pause(1e-17)
                '''
                #break
plt.show()
f.close()
