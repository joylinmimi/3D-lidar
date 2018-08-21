from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN
import pcl
import numpy as np
import matplotlib.cm as cm
#fig = plt.figure()
fig = plt.figure()
ax =Axes3D(fig)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

times=range(5)
secondloop=range(20)
for j in secondloop:
	c=len(times)+1
	array_x=[]
	array_y=[]
	array_z=[]
	for i in times:
		ax.clear()
		array=np.random.random((5, 3))
		print array
		array_x.extend(array[:,0])
		array_y.extend(array[:,1])
		array_z.extend(array[:,2])

		ax.scatter(array_x, array_y, array_z)
		c=c-1
		if c==1:
			ax.scatter(array_x, array_y, array_z)
			plt.pause(1)
		print j
		#print c,'c'
	
		
plt.show()
