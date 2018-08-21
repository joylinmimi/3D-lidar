import numpy as np
from matplotlib import pyplot as plt

from skimage.measure import ransac, LineModelND


np.random.seed(seed=1)

# generate coordinates of line
x = np.arange(-200, 200)
y = 0.2 * x + 20
data = np.column_stack([x, y])
print data
# add faulty data
#faulty = np.array(30 * [(180., -100)])
#faulty += 5 * np.random.normal(size=faulty.shape)
#data[:faulty.shape[0]] = faulty

# add gaussian noise to coordinates
#noise = np.random.normal(size=data.shape)
#data += 0.5 * noise
#data[::2] += 5 * noise[::2]
#data[::4] += 20 * noise[::4]

plt.plot(data[:, 0], data[:, 1], '.');
x = data[:, 0]
y = data[:, 1]

X = np.column_stack((x, np.ones_like(x)))

p, _, _, _ = np.linalg.lstsq(X, y)
p
m, c = p
plt.plot(x, y, 'b.')

xx = np.arange(-250, 250)
plt.plot(xx, m * xx + c, 'r-');
model = LineModelND()
model.estimate(data)
model.params
origin, direction = model.params
plt.plot(x, y, 'b.')
plt.plot(xx, model.predict_y(xx), 'r-');

model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                               residual_threshold=10, max_trials=1000)
outliers = (inliers == False)
yy = model_robust.predict_y(xx)
fig, ax = plt.subplots()
ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6, label='Inlier data')
ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6, label='Outlier data')
ax.plot(xx, yy, '-b', label='Robust line model')

plt.legend(loc='lower left')
plt.show()



