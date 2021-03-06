import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import np2pcd
import DBSCAN
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
	        array_x=[]
        	array_y=[]
        	array_z=[]
		
		count=0
		ax.clear() 
		#plt.pause(1e-17)
		filename_rgb="test_rgb.pcd"
		np2pcd.np2pcd(x_all, y_all, z_all, filename_rgb, rgb=True)
		
		stack_array=DBSCAN.DBScan()
		#print stack_array
		#print stack_array[1]
		
 		#ax.clear()
		ax.set_xlim3d([-3000,3000]) 
		ax.set_ylim3d([-3000,3000]) 
		ax.set_zlim3d([-500,500])
		ax.scatter(stack_array[0], stack_array[1], stack_array[2])
		#ax.scatter(xy2[:, 0], xy2[:, 1], xy2[:,2],color=col)
		#ax.plot(stack_array[0], stack_array[1], stack_array[2],'g.',markersize=4)
		plt.pause(1e-17)

		
plt.show()
f.close()

