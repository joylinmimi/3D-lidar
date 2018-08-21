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
for i in times:
	ax.clear()
	array=np.random.random((5, 3))
	ax.scatter(array[:, 0], array[:, 1], array[:,2])
	#ax.clear()
	plt.pause(1)
plt.show()
