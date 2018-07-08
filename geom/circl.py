# Local imports
from geom.pnt import Point
# Library imports
import math
import warnings
import random
import numpy as np


class Circle(Point):
    '''
    The Circle class is a child class of the Point. It includes a
        number of methods which can be used to calculate metrics between sets 
        of coordinates+radius such as intersections, etc.
    Note the difference in notation between:
        - 'circle' which at least represent a set of coordinates+radius
        - 'Circle' which is an instance of the class
    '''
    
    def __new__(cls, inputarray):
        '''
        A Circle can be created from a single set of coordinates or multiple:
        - single: a = Circle([5,7,1])
        - multiple: b = Circle([[5,7,1],[13,4,1]])  
        '''
        obj = np.asarray(inputarray).view(cls)
        try:
            obj - np.asarray([1, 1, 1])
            obj = obj.reshape(-1, 3)
            return obj
        except:
            raise ValueError("The input should have the shape of a (3,) or (*,3) array") 
    
    @classmethod
    def random(cls, range_xy, range_radius):
        '''
        Random instance of a Circle
        '''
        x = random.choice(range_xy)
        y = random.choice(range_xy)
        r = random.choice(range_radius)
        return cls([x, y, r])
    
    @classmethod
    def _reCircle(cls,circle_Circle_listOfCircles_listOfCircles):
        '''
        Wraps the class instantiation. To be used for:
            - allowing multiple input types in certain methods
            - instantiating certain method outputs as a Point
        '''
        # For read-continuity between class and instance methods
        Circle_out = cls(circle_Circle_listOfCircles_listOfCircles)
        return Circle_out
    
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
    
    
    def area(self):
        '''
        Returns: the area(s) as a numpy.ndarray
        '''
        return math.pi*self.r**2
      
    @staticmethod
    def _intersect(s_circle, m_circle, distance):
        d = distance
        r0 = s_circle.r
        r1 = m_circle.r
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
        summand_1 = s_circle.xy+a*(m_circle.xy-s_circle.xy)/d
        diff = m_circle.xy-s_circle.xy
        summand_2 = (h*(diff)/d)[:,::-1]
        intersects_1 = summand_1+summand_2*np.array([[-1,1]])
        intersects_2 = summand_1+summand_2*np.array([[1,-1]])
        return intersects_1, intersects_2
    
    def intersect(self, circle_or_list):
        '''
        Returns the intersecting point(s) with (an)other circle(s)
        Takes as input (a list of) coordinates or a (list of) Circle(s)
        '''
        m_circle = self._reCircle(circle_or_list)
        distance = self.distance(circle_or_list)
        intersects = self._intersect(self, m_circle, distance)
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