import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
x = np.arange(10)
ys = [i+x+(i*x)**2 for i in range(10)]


colors = itertools.cycle(["r", "b", "g"])
for y in ys:
    plt.scatter(x, y, color=next(colors))
plt.show()
