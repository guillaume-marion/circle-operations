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

Point.polyArea([sp1,sp2,sp3])
Point.polyArea([[6,12],[8,10],[6,10]])
### stability test ###
######################







############################
### Developing intersect ###
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
#plt.scatter(intersectpoints.x, intersectpoints.y)
for p in intersectpoints: #np.array(intersectpoints).reshape(-1,2):
    intplot = plt.scatter(p.x,p.y)
    ax.add_artist(intplot)
### Inspection ###
##################



p1 = Point([0,0])
p2 = Point([5,10])
p3 = Point([3,20])
p4 = Point([10,20])
p5 = Point([20,0])
p6 = Point([0,0])

def xRange(s_point, s_point_2):
    x_min = float(min(s_point.x, s_point_2.x))
    x_max = float(max(s_point.x, s_point_2.x))
    return x_min, x_max

def lineParameters(s_point, s_point_2):
    a = float((s_point_2.y-s_point.y)/(s_point_2.x-s_point.x))
    b = float(s_point.y - a*s_point.x)
    return a,b

def populate_line(s_point, s_point_2, nr_points, jitter_sd=1):
    a,b = lineParameters(s_point, s_point_2)
    x_min, x_max = xRange(s_point, s_point_2)
    x = np.random.uniform(x_min, x_max, nr_points)
    jitter_values = np.random.normal(0, jitter_sd, nr_points)
    y = (a*x) + b + jitter_values
    return np.array(list(zip(x,y)))

def populate_lines(m_point, nr_points, jitter_sd=1):
    populated_lines = np.array([]).reshape(0,2)
    nr_segments = len(m_point)
    for i in range(nr_segments-1):
        populated_line = populate_line(m_point[i], m_point[i+1], nr_points, jitter_sd)
        populated_lines = np.append(populated_line, populated_lines, axis=0)
    return populated_lines

result = populate_lines(Point([p1,p2,p3,p4,p5,p6]), nr_points=20) 
result = Point(result)

fig,ax = plt.subplots()
ax.set_xlim((0, 30))
ax.set_ylim((0, 30))
for c in result:
    cplot = plt.Circle((c.x, c.y), 2)
    ax.add_artist(cplot)
