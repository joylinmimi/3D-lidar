import numpy as np
def np2pcd(x, y, z, filename, rgb=False):
    rgb_value=0.05
    f=open(filename,'w')
    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\n")
    if (rgb==False):
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
    elif (rgb==True):
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
    f.write("WIDTH %d\n" % (x.size))
    f.write("HEIGHT 1\n")
    f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
    f.write("POINTS %d\n" % (x.size))
    f.write("DATA ascii\n")
    if (rgb==False):
        for i in range(np.size(x)):
            f.write("%.4f %.4f %.4f\n" % (x[i],y[i],z[i]))
    elif (rgb==True):
        for i in range(np.size(x)):
            f.write("%.4f %.4f %.4f %.4f\n" % (x[i],y[i],z[i], rgb_value))
    f.close()

