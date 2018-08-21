print(__doc__)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
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

# #############################################################################
# Generate sample data
centers = [[1, 1,1], [-1, -1,-1], [1, -1,0]]
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
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
    print my_members
    cluster_center = cluster_centers[k]
    #ax.scatter(X[my_members, 0], X[my_members, 1], col + '.')
    #ax.scatter(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             #markeredgecolor='k', markersize=14)
    ax.scatter(X[my_members, 0], X[my_members, 1],color=col,s=1)
    ax.scatter(cluster_center[0], cluster_center[1],color='black',s=1000)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
