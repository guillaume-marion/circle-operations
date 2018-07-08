# Local imports
from geom.pnt import Point
# Library imports
import math
import warnings
import random
import numpy as np


class Circle(Point):
    '''
    A Circle is constructed on top of a Point with a radius as supplementary parameter
    A Circle can be:
    - single: a = Circle([5,7,1])
    - multiple: b = Circle([[5,7,1],[13,4,1]])
    
    Calling x (,y or r) will always return the x (y, or r) values, either of the 
    single Circle or the multiple Circle (e.g. a.x == np.array([5]) and
                                                 b.x == np.array([5,13])
                                           )
    '''
    def __new__(cls, inputarray):
        obj = np.asarray(inputarray).view(cls)
        try:
            obj - np.asarray([1, 2, 3])
            obj = obj.reshape(-1, 3)
            return obj
        except:
            raise ValueError("Input should be a (3,) or (*,3) array") 
    
    @property
    def r(self):
        return np.asarray(self[:,2:])
    @r.setter
    def r(self, value):
        self[:,2:] = value
    @property
    def xy(self):
        return super(Circle, self).__new__(Point, self[:,:2])
    @xy.setter
    def xy(self, value):
        self[:,:2] = value

    @classmethod
    def toCircle(cls,circle_Circle_listOfCircles_listOfCircles):
        '''
        Simply wraps the instance creation for readability.
        Refer to Point.toPoint
        '''
        # For read-continuity between class and instance methods
        Circle_out = cls(circle_Circle_listOfCircles_listOfCircles)
        return Circle_out

    @classmethod
    def random(cls, range_xy, range_radius):
        x = random.choice(range_xy)
        y = random.choice(range_xy)
        r = random.choice(range_radius)
        return cls([x, y, r])
    
    @staticmethod
    def __area__(m_Circle):
        return math.pi*m_Circle.r**2
    
    def area(self):
        '''
        Returns the area of (a) circle(s)
        Can be applied on single or mutiple Circle, e.g.:
            Circle([4,5,1]).area()
            Circle([[4,5,1],[9,5,6]]).area()
        '''
        return self.__area__(self)
      
    @staticmethod
    def __intersect__(s_Circle, m_Circle, distance):
        d = distance
        r0 = s_Circle.r
        r1 = m_Circle.r
        inf_intersects = ((d==0) & (r0==r1))
        if inf_intersects.sum()>0:
            raise OverflowError('tangent circles')
        mask = ((d>r0+r1) | (d<abs(r0-r1)))
        if mask.sum()>0:
            warnings.warn('no intersection')
            if mask.sum()==len(r1):
                return None
        a = (r0**2-r1**2+d**2) / (2*d)
        a[mask]=np.nan
        sqrt_v = np.vectorize(math.sqrt)
        h = sqrt_v(r0**2-a**2)
        summand_1 = s_Circle.xy+a*(m_Circle.xy-s_Circle.xy)/d
        diff = m_Circle.xy-s_Circle.xy
        summand_2 = (h*(diff)/d)[:,::-1]
        intersects_1 = summand_1+summand_2*np.array([[-1,1]])
        intersects_2 = summand_1+summand_2*np.array([[1,-1]])
        return intersects_1, intersects_2
    
    def intersect(self, circle_or_list):
        '''
        Returns the intersecting point(s) with (an)other circle(s)
        Takes as input (a list of) coordinates or a (list of) Circle(s)
        '''
        m_Circle = self.toCircle(circle_or_list)
        distance = self.distance(circle_or_list)
        intersects = self.__intersect__(self, m_Circle, distance)
        return super(Circle, self).__new__(Point, intersects)
    
    ######################################################################
    ### WIP ##############################################################
    ###     ##############################################################
    ###     ##############################################################
    #         ############################################################
    ##       #############################################################
    ###     ##############################################################
    ####   ###############################################################
    ##### ################################################################
    ######################################################################
    
    def intersectCord(cls,circle_2):
        d = cls.distance(circle_2)
        r0 = cls.r
        r1 = circle_2.r
        a = (1/d)*math.sqrt((-d+r1-r0)*(-d-r1+r0)*(-d+r1+r0)*(d+r1+r0))
        return a
    
    def circularSegment(cls,cord):
        a = cord #self.intersectCord(circle_2)
        bR = cls.r
        sr = (1/2)*math.sqrt(4*bR**2-a**2)
        h = bR-sr
        bA = bR**2 * math.acos((bR-h)/bR) - (bR-h)*math.sqrt(2*bR*h-h**2)
        return bA
        
    def intersectArea(cls,circle_2,show_segments=False):
        a = cls.intersectCord(circle_2)
        A_self = cls.circularSegment(a)
        A_circle_2 = circle_2.circularSegment(a)
        A_total = A_self+A_circle_2
        if show_segments:
            return [A_total,[A_self,A_circle_2]]
        else:
            return A_total
    
    def includePoint(cls,point):
        px = point[0]
        py = point[1]
        value = (px-cls.x)**2+(py-cls.y)**2
        limit = (cls.r)**2
        result = value<limit or math.isclose(value,limit,abs_tol=1e-10)
        return result
    
    #@classmethod
    #def intersectAll(cls,circles_list,contained_by_all=True):
    
    def multiIntersectArea(cls,circles_list):
        # to do: Identify all intersection points 
        # > From those intersection points, identify those that are contained within all circles
        # If None, the circles do not all intersect into one area
        # If != None, calculate the intersecting Area based on "polygon of intersecting points" + "intersectCords"
        a=1