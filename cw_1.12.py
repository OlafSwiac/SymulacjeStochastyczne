import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def runisphere(n):
    points = []
    point = list(np.zeros(n))
    point_adj = list(np.zeros(n))
    for i in range(100000):
        R = 1.1
        while R > 1:
            for j in range(n):
                point[j] = np.random.rand() * 2 - 1
                point_adj[j] = point[j] ** 2
            R = sum(point_adj) ** (1/2)
        points.append(tuple([point_x/R for point_x in point]))
    return points


points = runisphere(3)
points_x1 = [point[0] for point in points]
points_x2 = [point[1] for point in points]
points_x3 = [point[2] for point in points]

fig = plt.figure()
ax = plt.axes(projection='3d')
my_cmap = plt.get_cmap('hsv')
ax.scatter3D(points_x1, points_x2, points_x3, c=(np.array(points_x1) + np.array(points_x2) + np.array(points_x3)), cmap=my_cmap)

"plt.hist(points_x1, bins=100, density=True, stacked=True)"
plt.show()

# rozklad brzegowy x1 -> jednostajny na (-1, 1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
hist, xedges, yedges = np.histogram2d(points_x1, points_x2, bins=100, range=[[-1, 1], [-1, 1]])

xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()

# rozklad brzegowy (x1, x2) -> jednostajny na kole o promieniu 1