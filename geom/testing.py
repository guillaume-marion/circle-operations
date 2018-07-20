from matplotlib import pyplot as plt

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
n_points = 2000
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





#####################
### angle between ###
p = Point([[8,8],[8,12],[12,12],[12,8]])
cp = Point([10,10])

def angleBetween(centerpoint, s_point, m_point):
    angleOrigin = 180-centerpoint.angleOffset(s_point)
    anglePoints = 180-centerpoint.angleOffset(m_point)
    anglePoints[anglePoints<angleOrigin] = anglePoints[anglePoints<angleOrigin ]+360
    return anglePoints-angleOrigin

i = 1
plt.scatter(p.x,p.y)
plt.scatter(cp.x,cp.y, c='red')
plt.scatter(p[i].x,p[i].y, marker='P', c='black', alpha=.5, s=250)
angleBetween(p[i], p.drop(i), cp)
Point.orderedIndex()
### angle between ###
#####################





###################
### outer bound ###
p1 = Point([5,7])
p2 = Point([15,7])
p3 = Point([25,22])
p4 = Point([5,7])
p = Point([p1,p2,p3,p4])
multic = Circle.populate_lines(p, nr_circles=15, radius_min=1, radius_max=4)

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
        intplot = plt.scatter(p.x,p.y, c='orange', alpha=0.5)
        ax.add_artist(intplot)
    else:
        intplot = plt.scatter(p.x,p.y, c='black')
        ax.add_artist(intplot)

l = all_intersects_l
mask = np.array([multic.encompass(intersec).any() for intersec in Point(l[0]).dropna()])
selection_outer_intersections = Point(l[0]).dropna()[mask==False,:]
plt.scatter(selection_outer_intersections.x,
            selection_outer_intersections.y,
            s=200, color='red', marker='P')

boundaries_l = multic.boundaries()
boundaries_points = [p[0] for p in boundaries_l]
boundaries_points = Point([item for sublist in boundaries_points for item in sublist])
boundaries_points = Point(np.unique(boundaries_points,axis=0))
boundaries_points  = boundaries_points.orderedPoints()
plt.plot(boundaries_points.x, boundaries_points.y, c='black')

### outer bound ###
###################




