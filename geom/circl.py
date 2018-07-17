# Local imports
#from geom.pnt import Point
# Library imports
import math
import warnings
import numpy as np


class Circle(Point):
    '''
    The Circle class is a child class of the Point. It includes a
        number of methods which can be used to calculate metrics between sets 
        of coordinates+radius such as intersections, etc.
    Note the difference in notation between:
        - 'circle' which at least represent a set of coordinates+radius
        - 'Circle' which represents the class
    '''
    
    def __new__(cls, inputarray):
        '''
        A Circle can be created from a single set of coordinates or multiple:
        - single: a = Circle([5,7,1])
        - multiple: b = Circle([[5,7,1],[13,4,1]])  
        '''
        obj = np.asarray(inputarray).view(cls)
        try:
            # Controlling for correct shape
            obj - np.asarray([1, 1, 1])
            obj = obj.reshape(-1, 3)
            return obj
        except:
            raise ValueError("The input should have the shape of a (3,) or (*,3) array") 
    
    @classmethod
    def _random(cls, x_min, x_max, y_min, y_max, radius_min, radius_max, nr_circles):
        '''
        Args:
            x_min: minium value for x-coordinates
            x_max: maximum value for x-coordinates
            y_min: minimum value for y-coordinates
            y_max: maximum value for y-coordinates
            radius_min: minimum value for radii
            radius_max: maximum value for radii
            nr_circles: number of xy-coordinates & radii to be produced
        
        Returns: 
            Random xy-coordinates & radii
        '''
        xy = super(Circle, cls)._random(x_min, x_max, y_min, y_max, nr_circles)
        r = np.random.uniform(radius_min, radius_max, nr_circles)
        xyr = np.dstack([xy,r])
        return xyr
    
    @classmethod
    def random(cls, x_min, x_max, y_min, y_max, radius_min, radius_max, nr_circles):
        '''
        Args:
            x_min: minium value for x-coordinates
            x_max: maximum value for x-coordinates
            y_min: minimum value for y-coordinates
            y_max: maximum value for y-coordinates
            radius_min: minimum value for radii
            radius_max: maximum value for radii
            nr_circles: number of xy-coordinates & radii to be produced
        
        Returns: 
            Random instance of a Circle
        '''
        xyr_values = cls._random(x_min, x_max, y_min, y_max, radius_min, radius_max, nr_circles)
        random_circle = cls(xyr_values)
        return random_circle
    
    @property
    def xy(self):
        return super(Circle, self).__new__(Point, self[:,:2])
    @xy.setter
    def xy(self, value):
        self[:,:2] = value
    @property
    def r(self):
        return np.asarray(self[:,2:])
    @r.setter
    def r(self, value):
        self[:,2:] = value    

    
    def area(self):
        '''
        Returns: the area(s) as a numpy.ndarray
        '''
        return math.pi*self.r**2
      
    @staticmethod
    def _intersect(s_circle_r, s_circle_xy, m_circle_r, m_circle_xy, distance):
        '''
        Args:
            s_circle_r: a single radius as a numpy.ndarray
            s_circle_xy: a single set of xy-coordinates as a numpy.ndarray
            m_circle_r: a single or multiple sets of radii as a numpy.ndarray
            m_circle_xy: a single or multiple sets of xy-coordinates as a
                numpy.ndarray
        
        Returns: the xy-coordinates of the intersections points between s_circle 
            and m_circle as a numpy.ndarray
        '''
        # Define necessary constants.
        d = distance
        r0 = s_circle_r
        r1 = m_circle_r
        xy0 = s_circle_xy
        xy1 = m_circle_xy
        # Raise for overlapping circles.
        inf_intersects = ((d==0) & (r0==r1))
        if inf_intersects.sum()>0:
            raise OverflowError('tangent circles')
        # Warn for non-intersecting circles.
        mask = ((d>r0+r1) | (d<abs(r0-r1)))
        if mask.sum()>0:
            warnings.warn('no intersection')
            # If only intersecting circles are provided, return None.
            if mask.sum()==len(r1):
                return None
        # Compute intersections in a vectorized fashion.
        a = (r0**2-r1**2+d**2) / (2*d)
        a[mask]=np.nan
        sqrt_v = np.vectorize(math.sqrt)
        h = sqrt_v(r0**2-a**2)
        summand_1 = xy0+a*(xy1-xy0)/d
        diff = xy1-xy0
        summand_2 = (h*(diff)/d)[:,::-1]
        intersect_1 = summand_1+summand_2*np.array([[-1,1]])
        intersect_2 = summand_1+summand_2*np.array([[1,-1]])
        return intersect_1, intersect_2
    
    def intersect(self, circle_or_list):
        '''
        Args:
            circle_or_list: a list or numpy array of single/multiple 
                xy-coordinates+radius or a (list of) Circle(s).
        
        Returns: the xy-coordinates of the intersections points between the
            origin circle and the (multiple) circle(s) as a Point
        '''
        # Extract the necessary parameters.
        s_circle_r = self.r
        s_circle_xy = self.xy
        m_circle = self._reClass(circle_or_list)
        m_circle_r = m_circle.r
        m_circle_xy = m_circle.xy
        distance = self.distance(circle_or_list)
        # Execute the static method with above parameters.
        intersects = self._intersect(s_circle_r=s_circle_r,
                                     s_circle_xy=s_circle_xy,
                                     m_circle_r=m_circle_r,
                                     m_circle_xy=m_circle_xy,
                                     distance=distance)
        # Change the type of the output to a Point.
        # We cannot use the _rePoint method as it will clash with Circle shape.
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