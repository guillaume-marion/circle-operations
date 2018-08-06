from geom.circl import Point, Circle
import matplotlib.pyplot as plt

mc = Circle([[9.9,7,2],
             [12,8,3],
             [5,7,3],
             [9,4,3],
             [11,12,2],
             [8,14,3],
             [6,10,1],
             [5.5,12,1.5],
             [8.5,10.7,0.5]
             ])

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.set_xlim((0, 20))
ax.set_ylim((0, 20))

for c in mc:
    cplot = plt.Circle((c.x,c.y), c.r, fill=False, color='gray', alpha=.7, linewidth=5)
    ax.add_artist(cplot)

mc.calc_intersections()
mc.calc_clusters()
cluster = mc.get_cluster(0)
cluster.calc_boundaries()

outer_b = Point([cluster.outer_boundaries[0][-1]]+cluster.outer_boundaries[0])
plt.scatter(outer_b.x, outer_b.y, color='blue', alpha=1, linewidth=5)
plt.plot(outer_b.x, outer_b.y, color='blue', alpha=.9, linewidth=3)

inner_b = Point([cluster.inner_boundaries[0][0][-1]]+cluster.inner_boundaries[0][0])
plt.scatter(inner_b.x, inner_b.y, color='green', alpha=1, linewidth=5)
plt.plot(inner_b.x, inner_b.y, color='green', alpha=.9, linewidth=3)

