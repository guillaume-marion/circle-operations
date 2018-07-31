from geom.circl import Circle, Point
from matplotlib import pyplot as plt
import numpy as np





######################
### stability test ###
# Test initiation
sp1 = Point([6,12])
sp2 = Point([8,10])
sp3 = Point([6,10])
mp1 = Point([sp1,sp2])
mp2 = Point([sp2,sp3])
# Test __repr__
sp1
mp1
# Test __str__
print(sp1)
print(mp1)
# Test setter
sp1.x=7
if (sp1 == Point([7,12])).sum()<2:
    raise ValueError
sp1 = Point([6,12])
mp2.x=np.array([[9],[7]])
if (mp2 == Point([[9,10],[7,10]])).sum()<4:
    raise ValueError
mp2 = Point([sp2,sp3])
# Test method
sp1.distance(sp2)
sp1.distance(mp2)
mp3 = Point([sp1,sp2,sp3])
### stability test ###
######################





#################
### Intersect ###
c1 = Circle([6,12,4])
c2 = Circle([10,12,4])
c3 = Circle([6,8,4])
c4 = Circle([10,8,4])
clist = [c1,c2,c3,c4]

fig,ax = plt.subplots()
ax.set_xlim((0, 20))
ax.set_ylim((0, 20))
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.xticks(np.arange(0, 21, 2.0))
plt.yticks(np.arange(0, 21, 2.0))
for c in clist:
    cplot = plt.Circle((c.x,c.y), c.r, color='blue', fill=False)
    ax.add_artist(cplot)

intersectpoints = c1.intersect(Circle(clist).drop(0))
for p in intersectpoints: #np.array(intersectpoints).reshape(-1,2):
    intplot = plt.scatter(p.x,p.y)
    ax.add_artist(intplot)
### Intersect ###
#################





#####################
### Encompassment ###
mc = Circle([c1,c2])

x_min = int(min(mc.x-max(mc.r)))-1
x_max = int(max(mc.x+max(mc.r)))+1
y_min = int(min(mc.y-max(mc.r)))-1
y_max = int(max(mc.y+max(mc.r)))+1
n_points = 1000
random_x = np.random.uniform(x_min, x_max, n_points).reshape(-1,1)
random_y = np.random.uniform(y_min, y_max, n_points).reshape(-1,1)
random_r = np.random.uniform(0.1, 0.5, n_points).reshape(-1,1)

fig,ax = plt.subplots()
fig.set_size_inches(15,10)
ax.set_xlim((x_min, x_max))
ax.set_ylim((y_min, y_max))
random_c = Circle([np.dstack([random_x, random_y, random_r])])
random_p = random_c.xy
for c in mc:
    cplot = plt.Circle((c.x, c.y), c.r, color='black', fill=False)
    ax.add_artist(cplot)
for c in random_c:
    if mc.encompass(c).any():
        circle_color = 'red'
        cplot = plt.Circle((c.x, c.y), c.r, color=circle_color, alpha=0.5)
        ax.add_artist(cplot)
    else:
        circle_color = 'blue'

for p in random_p:
    if mc.encompass(p).any():
        point_color = 'blue'
        pplot = plt.scatter(p.x, p.y, c=point_color, alpha=0.5)
        ax.add_artist(pplot)
    else:
        point_color = 'red'
### Encompassment ###
#####################





#######################
### Circle clusters ###
p1 = Point([5,6])
p2 = Point([6,7])
p = Point([p1,p2])
multic1 = Circle.populate_lines(p, nr_circles=15, radius_min=1, radius_max=4)
p3 = Point([15,7])
p4 = Point([16,6])
p = Point([p3,p4])
multic2 = Circle.populate_lines(p, nr_circles=15, radius_min=1, radius_max=4)
multic = Circle([multic1,multic2])
mpc = Circle([[25,4,2],[25,7,2],[25,10,2]])
multic = Circle(np.append(multic,mpc,axis=0))
c = Circle([10,20,2])
multic = Circle(np.append(c,multic,axis=0))
c = Circle([10,20,1])
multic = Circle(np.append(c,multic,axis=0))

multic.calc_intersections()
multic.calc_clusters()
multic._clusters_indices
result = [multic.get_cluster(_) for _ in range(multic.nr_clusters)]

fig,ax = plt.subplots()
fig.set_size_inches(15,10)
ax.set_xlim((0, 30))
ax.set_ylim((0, 30))
for i, resultc in enumerate(result):
    colors = ['black','red','blue','orange']
    for c in resultc:
        cplot = plt.Circle((c.x, c.y), c.r, color=colors[i], fill=False)
        ax.add_artist(cplot)
### Circle clusters ###
#######################






#########################################################
### correct index of intersections of cluster-results ###
p3 = Point([15,7])
p4 = Point([16,6])
p = Point([p3,p4])
multic = Circle.populate_lines(p, nr_circles=15, radius_min=1, radius_max=4)
multic.calc_intersections()
multic.calc_clusters()
some_cluster = multic.get_cluster(0)

for j in range(len(some_cluster)):
    for k in range(len(some_cluster)):
        if j==k:
            break
        fig,ax = plt.subplots()
        fig.set_size_inches(15,10)
        ax.set_xlim((0, 30))
        ax.set_ylim((0, 30))
        indices = [j, k]
        for i in indices:
            cplot = plt.Circle((some_cluster[i].x,some_cluster[i].y), some_cluster[i].r, fill=False)
            ax.add_artist(cplot)
        intercepts = some_cluster.intersections[indices[0]]
        intercepting_index = indices[1]-1 if indices[1]>indices[0] else indices[1]
        i1, i2 = intercepts[0][intercepting_index], intercepts[1][intercepting_index]
        plt.scatter(i1.x, i1.y)
        plt.scatter(i2.x, i2.y)
        filename = str(j)+str(k)
        #fig.savefig(filename)
### correct index of intersections of cluster-results ###
#########################################################





#####################
### angle between ###
p = Point([[8,8],[8,12],[12,12],[12,8]])
cp = Point([10,10])

i = 3
plt.scatter(p.x,p.y)
plt.scatter(cp.x,cp.y, c='red')
plt.scatter(p[i].x,p[i].y, marker='P', c='black', alpha=.5, s=250)
p.drop(i).orderedPoints(cp, return_angles=True)
p.drop(i).orderedPoints(cp, p[i], return_angles=True)
### angle between ###
#####################





#######################################
### outer bound - multiple clusters ###

# No inner boundary hole
p1 = Point([5,25])
p2 = Point([6,30])
p = Point([p1,p2]) 
multic1 = Circle.populate_lines(p, nr_circles=30, radius_min=1, radius_max=2)

# 1 inner boundary hole
p1 = Point([25,5])
p2 = Point([26,15])
p3 = Point([30,15])
p4 = Point([31,5])
p5 = Point([25,5])
p = Point([p1,p2,p3,p4,p5]) 
multic2 = Circle.populate_lines(p, nr_circles=30, radius_min=1, radius_max=2)

# 3 inner boundary holes # Here resides an error
#  I think it is because we take least angle
#  Maybe in case of inner boundaries, we should take max angle
p1 = Point([10,7])
p2 = Point([20,7])
p3 = Point([10,22])
p4 = Point([20,22])
p5 = Point([10,7])
p6 = Point([5,14])
p7 = Point([10,22])
p = Point([p1,p2,p3,p4,p5,p6,p7]) 
multic3 = Circle.populate_lines(p, nr_circles=50, radius_min=1, radius_max=2, jitter_sd=0.01)

multic = Circle(np.append(np.append(multic1,multic2, axis=0), multic3, axis=0))

fig,ax = plt.subplots()
fig.set_size_inches(15,10)
ax.set_xlim((0, 35))
ax.set_ylim((0, 35))
for c in multic:
    cplot = plt.Circle((c.x, c.y), c.r, color='blue', fill=False, alpha=.5)
    ax.add_artist(cplot)

multic.calc_intersections()
multic.calc_clusters()
for i in range(multic.nr_clusters):
    cluster = multic.get_cluster(i)
    cluster.calc_boundaries()
    
    ordered_boundaries_p, ordered_boundaries_cp,_ = cluster.outer_boundaries
    ordered_boundaries_pp = Point(ordered_boundaries_p)
    plt.scatter(ordered_boundaries_pp.x, ordered_boundaries_pp.y, color='black')
    ordered_boundaries = np.hstack((ordered_boundaries_cp, ordered_boundaries_p))
    ordered_boundaries = Point(ordered_boundaries)
    plt.plot(ordered_boundaries.x, ordered_boundaries.y, c='black')
    for inner_boundaries in cluster.inner_boundaries:
        ordered_boundaries = Point(inner_boundaries)
        plt.plot(ordered_boundaries.x, ordered_boundaries.y, c='green')
        plt.scatter(ordered_boundaries.x, ordered_boundaries.y, c='green')
### outer bound - multiple clusters ###
#######################################
               




    