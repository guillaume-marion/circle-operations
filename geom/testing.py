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





