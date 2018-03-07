# Local imports
from geom.pnt import ppoint
# Library imports
import math
import warnings
import numpy as np
from itertools import chain


class ccircle(ppoint):
    '''
    A ccircle is constructed on top of a ppoint with a radius as supplementary parameter
    A ccircle can be:
    - single: a = ccircle([5,7,1])
    - multiple: b = ccircle([[5,7,1],[13,4,1]])
    
    Calling x (,y or r) will always return the x (y, or r) values, either of the 
    single ccircle or the multiple ccircle (e.g. a.x == np.array([5]) and
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
        return super(ccircle, self).__new__(ppoint, self[:,:2])
        #return np.asarray(self[:,:2])
    @xy.setter
    def xy(self, value):
        self[:,:2] = value

    @classmethod
    def toCircle(cls,circle_ccircle_listOfCircles_listOfCcircles):
        '''
        Simply wraps the instance creation for readability.
        Refer to ppoint.toPoint
        '''
        # For read-continuity between class and instance methods
        ccircle_out = cls(circle_ccircle_listOfCircles_listOfCcircles)
        return ccircle_out
    
    @staticmethod
    def __area__(m_ccircle):
        return math.pi*m_ccircle.r**2
    
    def area(self):
        '''
        Returns the area of (a) circle(s)
        Can be applied on single or mutiple ccircle, e.g.:
            ccircle([4,5,1]).area()
            ccircle([[4,5,1],[9,5,6]]).area()
        '''
        return self.__area__(self)
      
    @staticmethod
    def __intersect__(s_ccircle, m_ccircle, distance):
        d = distance
        r0 = s_ccircle.r
        r1 = m_ccircle.r
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
        summand_1 = s_ccircle.xy+a*(m_ccircle.xy-s_ccircle.xy)/d
        diff = m_ccircle.xy-s_ccircle.xy
        summand_2 = (h*(diff)/d)[:,::-1]
        intersects_1 = summand_1+summand_2*np.array([[-1,1]])
        intersects_2 = summand_1+summand_2*np.array([[1,-1]])
        return intersects_1, intersects_2
    
    def intersect(self, circle_or_list):
        '''
        Returns the intersecting point(s) with (an)other circle(s)
        Takes as input (a list of) coordinates or a (list of) ccircle(s)
        '''
        m_ccircle = self.toCircle(circle_or_list)
        distance = self.distance(circle_or_list)
        return self.__intersect__(self, m_ccircle, distance)

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
    
    def polygonArea(point_list):
        # Area of 3 or more points
        a=1
    
    def multiIntersectArea(cls,circles_list):
        # to do: Identify all intersection points 
        # > From those intersection points, identify those that are contained within all circles
        # If None, the circles do not all intersect into one area
        # If != None, calculate the intersecting Area based on "polygon of intersecting points" + "intersectCords"
        a=1