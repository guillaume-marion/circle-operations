from matplotlib import pyplot as plt

######################
### stability test ###
# Test initiation
sp1 = ppoint([6,12])
sp2 = ppoint([8,10])
sp3 = ppoint([6,10])
mp1 = ppoint([sp1,sp2])
mp2 = ppoint([sp2,sp3])
# Test __repr__
sp1
mp1
# Test __str__
print(sp1)
print(mp1)
# Test setter
sp1.x=7
if (sp1 == ppoint([7,12])).sum()<2:
    raise ValueError
sp1 = ppoint([6,12])
mp2.x=np.array([[9],[7]])
if (mp2 == ppoint([[9,10],[7,10]])).sum()<4:
    raise ValueError
mp2 = ppoint([sp2,sp3])
# Test method
sp1.distance(sp2)
sp1.distance([sp2])
sp1.distance([5,4])
sp1.distance([mp2])
sp1.distance([[6,5],[9,2]])

ppoint.polyArea([sp1,sp2,sp3])
ppoint.polyArea([[6,12],[8,10],[6,10]])
### stability test ###
######################






############################
### Developing intersect ###

c1 = ccircle([6,12,2])
c2 = ccircle([8,12,2])
c3 = ccircle([6,10,2])
c4 = ccircle([8,10,2])
clist = [c1,c2,c3,c4]

ct = ccircle([6,40,1])
compare = [c2,c3,c4,ct]
ccompare = ccircle.toCircle(compare)
d = c1.distance(compare)
r0 = c1.r
r1 = ccompare.r

inf_intersects = ((d==0) & (r0==r1))
#if inf_intersects.sum()>0:
#    raise OverflowError('overlapping circles > infinite number of solutions')
mask = ((d>r0+r1) | (d<abs(r0-r1)))
#if mask.sum()>0:
#    warnings.warn('no intersection > returning None')
#    if mask.sum()==len(r1):
#        return None
    
a = (r0**2-r1**2+d**2) / (2*d)
a[mask]=np.nan
sqrt_v = np.vectorize(math.sqrt)
h = sqrt_v(r0**2-a**2)

summand_1 = c1.xy+a*(ccompare.xy-c1.xy)/d
diff = ccompare.xy-c1.xy
summand_2 = (h*(diff)/d)[:,::-1]
intersects_1 = summand_1+summand_2*np.array([[-1,1]])
intersects_2 = summand_1+summand_2*np.array([[1,-1]])
type(intersects_1)

### Developing intersect ###
############################





##################
### Inspection ###
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
for p in np.array(intersectpoints).reshape(-1,2):
    plt.scatter(p[0],p[1])
### Inspection ###
##################


