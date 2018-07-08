# Library imports
import math
import random
import numpy as np


# Defining the Point class
class Point(np.ndarray):
    '''
    The Point class is a child class of the numpy.ndarray. It includes a
        number of methods which can be used to calculate metrics between sets 
        of coordinates such as distances, etc.
    Note the difference in notation between:
        - 'point' which at least represent a set of coordinates
        - 'Point' which is an instance of the class
    '''
    
    def __new__(cls, inputarray):
        '''
        A Point can be created from a single set of coordinates or multiple:
        - single: a = Point([5,7])
        - multiple: b = Point([[5,7],[13,4]])  
        '''
        obj = np.asarray(inputarray).view(cls)
        try:
            # Controlling for correct shape
            obj - np.asarray([1, 1])
            obj = obj.reshape(-1, 2)
            return obj
        except:
            raise ValueError("The input should have the shape of a (2,) or (*,2) array.") 
    
    @classmethod
    def random(cls, range_xy):
        '''
        Random instance of a Point
        '''
        x = random.choice(range_xy)
        y = random.choice(range_xy)
        return cls([x, y])
    
    @classmethod
    def _rePoint(cls, something_to_instantiate):
        '''
        Wraps the class instantiation. To be used for:
            - allowing multiple input types in certain methods
            - instantiating certain method outputs as a Point
        '''
        point_out = cls(something_to_instantiate)
        return point_out
    
    @property
    def x(self):
        '''
        Calling x/y will always return the x/y value(s), either of the 
            single Point or the multiple Point (e.g. a.x == np.array([5]) and
                                                     b.x == np.array([5,13])).
        '''
        return np.asarray(self[:,:1])
    
    @x.setter
    def x(self, value):
        '''
        See comment for x property.
        '''
        self[:,:1] = value
        
    @property
    def y(self):
        '''
        See comment for x property.
        '''
        return np.asarray(self[:,1:2])
    
    @y.setter
    def y(self, value):
        '''
        See comment for x property.
        '''
        self[:,1:2] = value
        
    def __getitem__(self,val):
        '''
        If val is an int, then consider it as a row-selection which should
            result in a Point instance. If a slice, then use numpy's 
            __getitem__ method.
        Typical usecase: selecting a single row from a multiple Point instance
            should also result in a Point with the correct shape to allow
            use of the class' methods.
        '''
        if type(val)==int:
            return self._rePoint(np.asarray(self)[val])
        else:
            return super(Point, self).__getitem__(val)
    
    def __str__(self):
        '''
        Uses numpy's __str__ method.
        '''
        return super(Point, self).__str__()
    
    def __repr__(self):
        '''
        Defines custom __repr__ method.
        '''
        return (str(self.__class__.__name__)+"([\n {:>10}\n])"
                .format(self.__str__()[1:-1]))
    
    @staticmethod
    def _distance(s_point, m_point):
        '''
        Args:
            s_point: a single set of xy-coordinates as a numpy.ndarray
            m_point: a single or multiple sets of xy-coordinates as a
                numpy.ndarray
        
        Returns: The euclidean distance(s) between the s_point and m_point as 
            a numpy.ndarray
        '''
        return np.sqrt((m_point.x-s_point.x)**2+(m_point.y-s_point.y)**2)
    
    def distance(self, point_or_list):
        '''
        Args:
            point_or_list: a list or numpy array of single/multiple 
                xy-coordinates or a (list of) Point(s).
        
        Returns: The euclidean distance(s) to the origin and the (multiple) 
            point(s) as a numpy.ndarray.
        '''
        m_point = self._rePoint(point_or_list) 
        return self._distance(self, m_point)
    
    @staticmethod
    def _angleOffset(s_point, m_point):
        '''
        Args:
            s_point: a single set of xy-coordinates as a numpy.ndarray
            m_point: a single or multiple sets of xy-coordinates as a
                numpy.ndarray
        
        Returns: the angle at which the horizontal line needs to rotate
            clockwise in order to match the line between the s_point and
            the m_point.
        '''
        atan2_v = np.vectorize(math.atan2)
        degrees_v = np.vectorize(math.degrees)  
        dx = m_point.x - s_point.x
        dy = m_point.y - s_point.y
        return degrees_v(atan2_v(dy, dx))
    
    def angleOffset(self, point_or_list):
        '''
        Args:
            point_or_list: a list or numpy array of single/multiple 
                xy-coordinates or a (list of) Point(s).
        
        Returns: the angle at which the horizontal line needs to rotate
            clockwise in order to match the line between the origin and
            the (multiple) point(s) as a numpy.ndarray.
        '''
        m_point = self._rePoint(point_or_list) 
        return self._angleOffset(self, m_point)
    
    @staticmethod
    def _centroid(m_point):
        '''
        Args:
            m_point: a single or multiple sets of xy-coordinates as a
                numpy.ndarray
        
        Returns: the xy-coordinates of the centroid as a numpy.ndarray
        '''
        n_points = len(m_point)
        centroid = [m_point.x.sum() / n_points, m_point.y.sum() / n_points]
        return centroid
    
    @classmethod
    def centroid(cls, point_or_list):
        '''
        Args:
            point_or_list: a list or numpy array of single/multiple 
                xy-coordinates or a (list of) Point(s).
        
        Returns: the xy-coordinates of the centroid as a Point
        '''
        m_point = cls._rePoint(point_or_list) 
        centroid = cls._centroid(m_point)
        centroid_ppoint = cls._rePoint(centroid)
        return centroid_ppoint
    
    @classmethod
    def orderedIndex(cls, point_or_list):
        '''
        Args:
            point_or_list: a list or numpy array of single/multiple 
                xy-coordinates or a (list of) Point(s).
        
        Returns: the index of the points in a clockwise order as a numpy.ndarray
        '''
        centr = cls.centroid(point_or_list)
        #return points_list[centr.angleOffset(points_list).argsort()][::-1]
        return (centr.angleOffset(point_or_list)*-1).argsort(axis=0)

    @staticmethod
    def _polyArea(m_point_ordered):
        '''
        Args:
            m_point_ordered: multiple sets of xy-coordinates in clockwise order
                as a numpy.ndarray
        
        Returns: the area of the polygon bounded by the points as a numpy.ndarray
        '''
        return 0.5*np.abs(np.dot(m_point_ordered.x.T[0],np.roll(m_point_ordered.y.T[0],1))
                          -np.dot(m_point_ordered.y.T[0],np.roll(m_point_ordered.x.T[0],1)))
        
    @classmethod
    def polyArea(cls, point_or_list):
        '''
        Args:
            point_or_list: a list or numpy array of single/multiple 
                xy-coordinates or a (list of) Point(s).
        
        Returns: the area of the polygon bounded by the points as a numpy.ndarray
        '''
        m_point = cls._rePoint(point_or_list)
        m_point_ordered = m_point[cls.orderedIndex(point_or_list)][:,0]
        area = cls._polyArea(m_point_ordered)
        return area
    