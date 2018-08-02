from geom.circl import Circle, Point
from matplotlib import pyplot as plt
import numpy as np





######################
### stability test ###
from geom.circl import Circle, Point

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
from geom.circl import Circle, Point

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
from geom.circl import Circle, Point
    
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
from geom.circl import Circle, Point

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
from geom.circl import Circle, Point

plt.ioff()

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
        fig.savefig(filename)
        plt.close(fig)
        
plt.ion()
### correct index of intersections of cluster-results ###
#########################################################





#####################
### angle between ###
from geom.circl import Circle, Point

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
from geom.circl import Circle, Point

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

# 3 inner boundary holes 
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

multic.get_cluster(0)

for i in range(multic.nr_clusters):
    cluster = multic.get_cluster(i)
    cluster.calc_boundaries()
    ordered_boundaries_p, ordered_boundaries_c, = cluster.outer_boundaries
    ordered_boundaries = Point(ordered_boundaries_p)
    plt.scatter(ordered_boundaries.x, ordered_boundaries.y, color='black')
    plt.plot(ordered_boundaries.x, ordered_boundaries.y, c='black')
    for inner_boundaries,_ in cluster.inner_boundaries:
        ordered_boundaries = Point(inner_boundaries)
        plt.plot(ordered_boundaries.x, ordered_boundaries.y, c='green')
        plt.scatter(ordered_boundaries.x, ordered_boundaries.y, c='green')
### outer bound - multiple clusters ###
#######################################





#########################
### Polygon encompass ### 
poly = Point([[5,5],[5,10],[7.5,7.5],[10,10],[10,8],[8,6],[10,5],[5,5]])
multip = Point.random(4,11,4,11, 5000)
for p in multip:
    isin = poly.polyEncompass(p)
    if isin:
        plt.scatter(p.x, p.y, color="green")
    else:
        plt.scatter(p.x, p.y, color="orange")
### Polygon encompass ### 
#########################
        
        
        
        
        
#####################################
### outer bound - random clusters ###
from geom.circl import Circle, Point

multic = Circle.random(5,30,5,30,1,3,50)

fig,ax = plt.subplots()
fig.set_size_inches(15,10)
ax.set_xlim((0, 35))
ax.set_ylim((0, 35))
#for c in multic:
#    cplot = plt.Circle((c.x, c.y), c.r, color='black', fill=False, alpha=.5)
#    ax.add_artist(cplot)

multic.calc_intersections()
multic.calc_clusters()

for i in range(multic.nr_clusters):
    cluster = multic.get_cluster(i)
    cluster.calc_boundaries()
    
    if len(cluster)==1:
        c = cluster[0]
        cplot = plt.Circle((c.x, c.y), c.r, color='blue', fill=True, alpha=.25)
        ax.add_artist(cplot)
        
    if len(cluster)==2:
        for c in cluster:
            cplot = plt.Circle((c.x, c.y), c.r, color='orange', fill=True, alpha=.25)
            ax.add_artist(cplot)
        
    if len(cluster)>2:
        # For every cluster...
        for c in cluster:
            cplot = plt.Circle((c.x, c.y), c.r, color='green', fill=True, alpha=.25)
            ax.add_artist(cplot)
            
        # Get the boundaries & Circles. 
        ordered_b, ordered_c, = cluster.outer_boundaries
        # Close the loop.
        ordered_all = Point([ordered_b[-1]]+ordered_b)
        # Scatter boundaries and plot segments in dotted lines.
        plt.scatter(ordered_all.x, ordered_all.y, color='black')
        plt.plot(ordered_all.x, ordered_all.y, c='black', ls='--')
        # Get the centerpoints and test for encompassment.
        ordered_cp = [_.xy for _ in ordered_c]
        cp_in_polygon = np.array([ordered_all.polyEncompass(_) for _ in ordered_cp])
        cp_notin_polygon = cp_in_polygon==False
        # Create the segments (i,i+1) from the closed loop boundaries.
        segments = [ordered_all[[i,i+1]] for i in range(len(ordered_all)-1)]
        
        # Supplementary test for encompassment !!!!!
        centroids = Point([_.centroid() for _ in segments])
        distances = [float(ordered_cp[i].distance(_)) for i,_ in enumerate(centroids)]
        min_distances = [_.distance(centroids).min() for _ in ordered_cp]
        cp_on_correct_side = np.array(distances)<=np.array(min_distances)
        
        mask = (cp_notin_polygon) & (cp_on_correct_side)
        
        
        # Plot the centerpoint and full segment-line if the centerpoint is not encompassed.
        for i,test in enumerate(mask):
            if test:
                plt.scatter(ordered_cp[i].x, ordered_cp[i].y, c='black', marker='P')
                plt.plot(segments[i].x, segments[i].y, c='black')
        
        # For every hole in the cluster...
        for inner_boundary in cluster.inner_boundaries:
            # Get the boundaries & Circles. 
            ordered_b, ordered_c, = inner_boundary
            # Close the loop.
            ordered_all = Point([ordered_b[-1]]+ordered_b)
            # Scatter boundaries and plot segments in dotted lines.
            plt.scatter(ordered_all .x, ordered_all .y, c='green')
            plt.plot(ordered_all .x, ordered_all .y, c='green', ls='--')
            # Get the centerpoints and test for encompassment.
            ordered_cp = [_.xy for _ in ordered_c]
            cp_in_polygon = [ordered_all.polyEncompass(_) for _ in ordered_cp]
            # Create the segments (i,i+1) from the closed loop boundaries.
            segments = [ordered_all[[i,i+1]] for i in range(len(ordered_all)-1)]
            # Plot the centerpoint and full segment-line if the centerpoint is not encompassed.
            for i,test in enumerate(cp_in_polygon):
                if test:
                    plt.scatter(ordered_cp[i].x, ordered_cp[i].y, c='green', marker='_')
                    plt.plot(segments[i].x, segments[i].y, c='green')
### outer bound - random clusters ###
##################################### 





#######################################
### Testing cluster area vs mc area ### 
from geom.circl import Circle, Point

# Square without inner hole
mc = Circle([[4,6,2],
             [6,8,2],
             [8,6,2],
             [6,4,2],
             [6,6,2]])
    
fig,ax = plt.subplots()
fig.set_size_inches(15,10)
ax.set_xlim((0, 35))
ax.set_ylim((0, 35))
for c in mc:
    cplot = plt.Circle((c.x, c.y), c.r, color='blue', fill=False, alpha=.5)
    ax.add_artist(cplot)


mc.calc_intersections()
mc.calc_clusters()
cluster = mc.get_cluster(0)
cluster.calc_boundaries()

# Deduction
a = ((ordered_boundaries_p[0].distance(ordered_boundaries_p[1])**2)+
     (mc[:-1].area()/2).sum())
# Simulation
b = cluster.mcArea(300000)
# Exact
c = cluster.flatArea()

print(a,b,c)
### Testing cluster area vs mc area ### 
#######################################







