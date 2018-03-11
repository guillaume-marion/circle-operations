

import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain




class ppoint:
	'''
	This is the first trial where the ppoint class was not built on top of the numpy.ndarray.
	Downstream it required a lot of map() usage forcing the re-evaluation of the class' basic structure.
	'''
    
    def __init__(self,x_coordinate,y_coordinate):
        self.x = x_coordinate
        self.y = y_coordinate
        
    def __str__(self):
        return str((self.x,self.y))
      
    def sDistance(cls,point_2):
        '''
        Distance to a single ppoint 
        '''
        return math.sqrt( (point_2.x-cls.x)**2 + (point_2.y-cls.y)**2 )
    
    def mDistance(cls,points_list):
        '''
        Distance to a list of ppoint 
        '''
        map_result = map(cls.sDistance,points_list)
        return list(map_result)
    
    def distance(cls,point_2_or_points_list):
        '''
        Takes a single ppoint or a list of ppoint as input
        Oututs the distance to the center(s) of a (a list of) ppoint(s)
        '''
        if type(point_2_or_points_list)==list:
            return(cls.mDistance(point_2_or_points_list))
        if type(point_2_or_points_list)==type(cls):
            return(cls.sDistance(point_2_or_points_list))
    
    def findPoint(points_list,method='top'):
        if method='top':
            
        
    def orderPoints(points_list):
        # Order points clockwise
        a=1
    


class ccircle(ppoint):
    
    def __init__(self,x_coordinate,y_coordinate,radius):
        super().__init__(x_coordinate,y_coordinate)
        self.r = radius
        
    def __str__(self):
        return str((self.x,self.y,self.r))
        
    def area(cls):
        '''
        Computes the area of a circle
        '''
        A = math.pi*cls.r**2
        return A
      
    def sIntersect(cls,circle_2):
        '''
        Intersect with a single ccircle
        '''
        d = cls.distance(circle_2)
        r0 = cls.r
        r1 = circle_2.r
        if (d>r0+r1) | (d<abs(r0-r1)) :
            warnings.warn('no intersection > returning None')
            return None
        if (d==0) & (r0==r1):
            raise OverflowError('overlapping circles > infinite number of solutions')
        a = (r0**2-r1**2+d**2) / (2*d)
        h = math.sqrt(r0**2-a**2)
        P0 = np.array([cls.x,cls.y])
        P1 = np.array([circle_2.x,circle_2.y])
        P2 = P0+a*(P1-P0)/d
        x_y_diff = P1-P0
        substra_add_end = h*(x_y_diff)/d
        I1 = P2+substra_add_end[::-1]*[1,-1]
        I2 = P2+substra_add_end[::-1]*[-1,1]
        return I1,I2
    
    def mIntersect(cls,circles_list,flat=True):
        '''
        Intersect with a list of ccircles
        '''
        map_result = map(cls.sIntersect, circles_list)
        if flat:
            return list(chain.from_iterable(map_result))
        else:
            return list(map_result)
       
    def intersect(cls,circle_2_or_circles_list,flat=True):
        '''
        Takes a single ccricle or a list of ccirles as input
        Outputs the coordinate(s) of the intersection point(s)
        '''
        if type(circle_2_or_circles_list)==list:
            return(cls.mIntersect(circle_2_or_circles_list, flat=flat))
        if type(circle_2_or_circles_list)==type(cls):
            return(cls.sIntersect(circle_2_or_circles_list))

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


