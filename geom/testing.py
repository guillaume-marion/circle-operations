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
sp1.distance([sp2])
sp1.distance([5,4])
sp1.distance([mp2])
sp1.distance([[6,5],[9,2]])

mp3 = Point([sp1,sp2,sp3])
mp3.polyArea()
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

intersectpoints = c1.intersect(clist[1:])
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

result = multic.cluster()

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





###################
### outer bound ###
p1 = Point([5,7])
p2 = Point([15,7])
p3 = Point([10,22]) # 25,22
p4 = Point([5,7])
p = Point([p1,p2,p3,p4])
multic = Circle.populate_lines(p, nr_circles=15, radius_min=1, radius_max=3)

fig,ax = plt.subplots()
fig.set_size_inches(15,10)
ax.set_xlim((0, 30))
ax.set_ylim((0, 30))
for c in multic:
    cplot = plt.Circle((c.x, c.y), c.r, color='blue', fill=False)
    ax.add_artist(cplot)

all_intersects_l = multic.intersections()
all_intersects = Point(all_intersects_l).dropna()

for p in all_intersects:
    if multic.encompass(p).any():
        a = 1
        intplot = plt.scatter(p.x,p.y, c='orange', alpha=0.3)
        ax.add_artist(intplot)
    else:
        intplot = plt.scatter(p.x,p.y, c='black')
        ax.add_artist(intplot)

ordered_boundaries_p, ordered_boundaries_cp, ordered_angles, non_selected, _ = multic.orderedBoundaries()
ordered_boundaries = np.hstack((ordered_boundaries_cp, ordered_boundaries_p))
ordered_boundaries = ordered_boundaries.reshape(-1,2)
ordered_boundaries = Point([ordered_boundaries])
plt.plot(ordered_boundaries.x, ordered_boundaries.y, c='black')

non_selected = Point([item for sublist in non_selected for item in sublist])
plt.scatter(non_selected.x, non_selected.y, c='red', marker='P', s=200)


  ###########
 ### WIP ###
##########
 
# TO FIND THE INNER BOUNDARY SIMPLY RE-EXECUTE THE METHOD ON THE REMAINING BOUNDARIES
### REWRITE METHOD TO TAKE AS ARGUMENT : BOUNDARIES, CP (such that it can be looped)
### TEST METHOD FOR MULTIPLE HOLES
ordered_boundaries_p, ordered_boundaries_cp, ordered_angles, non_selected, _ = multic.orderedBoundaries()
boundaries_l, centerpoints_l = non_selected, _ 
prec=8

# Params
first = True
ordered_boundaries_l = []
ordered_centerpoints_l = []
ordered_angles_l = []
remaining_boundaries_l = []
remaining_centerpoints_l = []

# Starting point = of the Circles with the leftmost intersections, we select
#  that Circle which also has the upmost intersection. From that Circle we
#  take the leftmost intersection as starting point.
xmin = min([_.x.min().round(prec) for _ in boundaries_l])
xmin_index = [i for i,_ in enumerate(boundaries_l) if _.x.min().round(prec) == xmin]
ymax_xmin = [boundaries_l[_].y.max().round(prec) for _ in xmin_index]
xmin_subindex = [i for i,_ in enumerate(ymax_xmin) if _ == max(ymax_xmin)][0]
xmin_index = xmin_index[xmin_subindex]
previous_i = xmin_index
boundary = [_ for _ in boundaries_l[xmin_index] if _.x.round(prec) == xmin][0]
cp = centerpoints_l[xmin_index]

while True: 
    # Favor new centerpoint
    i = ([i for i,_ in enumerate(boundaries_l) 
          if np.any(_.round(prec)==boundary.round(prec))])
    i = [_ for _ in i if _ != previous_i][0] if len(i)>1 else i[0]
    if first:
        i = previous_i
    bp = boundaries_l[i]
    cp = centerpoints_l[i]
    n_bounds = len(bp)
    # Find the index of the boundary in the bp's
    mask = (bp.round(prec)==boundary.round(prec)).all(axis=1)
    j = int(np.arange(n_bounds)[mask])
    # Order the bp's excluding the previous boundary
    ordered_remainders, angles = bp.drop(j).orderedPoints(cp, bp[j], True)
    angle = min(angles)
    # The next boundary is found by taking the first bp of the ordered bp's
    boundary = ordered_remainders[0]
    # Break if boundary is closed
    if not first and np.all(ordered_boundaries_l[0].round(prec)==boundary.round(prec)):
        break
    # Add it to the list as well as the corresponding cp
    ordered_boundaries_l.append(boundary)
    ordered_centerpoints_l.append(cp)
    ordered_angles_l.append(angle)
    # Reset previous i
    previous_i = i
    first = False
# Collect boundaries which have note been used in the closed circuit, 
#  i.e. boundaries of 'inner holes'
try:
    indices, candidates = zip(*[(i,p) for (i,sublist) in enumerate(boundaries_l) for p in sublist 
                                if (p.round(prec) != np.array(ordered_boundaries_l).round(prec)).all()])
    remaining_centerpoints = Point([centerpoints_l[_] for _ in indices]).round(prec) 
    for u_cp in np.unique(remaining_centerpoints, axis=0):
        mask = (remaining_centerpoints == u_cp).all(axis=1)
        p = Point([candidates])[mask]
        cp = Point([u_cp])
        remaining_boundaries_l.append(p)
        remaining_centerpoints_l.append(cp)
except ValueError:
    warnings.warn('no inner boundaries')

ordered_boundaries = Point([ordered_boundaries_l])
plt.plot(ordered_boundaries.x, ordered_boundaries.y, c='black')
### outer bound ###
###################

