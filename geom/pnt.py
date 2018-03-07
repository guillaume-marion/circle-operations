# Library imports
import math
import warnings
import random
import numpy as np



class ppoint(np.ndarray):
    '''
    A ppoint can be:
    - single: a = ppoint([5,7])
    - multiple: b = ppoint([[5,7],[13,4]])
    
    Calling x (or y) will always return the x (or y) values, either of the 
    single ppoint or the multiple ppoint (e.g. a.x == np.array([5]) and
                                              b.x == np.array([5,13])
                                         )
    '''
    def __new__(cls, inputarray):
        obj = np.asarray(inputarray).view(cls)
        try:
            obj - np.asarray([1, 2])
            obj = obj.reshape(-1, 2)
            return obj
        except:
            raise ValueError("Input should be a (2,) or (*,2) array") 
    
    @property
    def x(self):
        return np.asarray(self[:,:1])
    @x.setter
    def x(self, value):
        self[:,:1] = value
    @property
    def y(self):
        return np.asarray(self[:,1:2])
    @y.setter
    def y(self, value):
        self[:,1:2] = value
        
    def __str__(self):
        return str(self.__class__.__name__)+"(\n{:>10}\n)".format(repr(self))
    
    def __repr__(self):
        return str(np.array(self))
    
    @classmethod
    def toPoint(cls,point_ppoint_listOfPoints_listOfPPoints):
        '''
        Simply wraps the instance creation for readability.
        To be used in most methods for user easy-of-use, e.g.:
            type(toPoint(ppoint)) == __main__.ppoint
            type(toPoint([5,7])) == __main__.ppoint
            type(toPoint([ppoint_1,ppoint_2])) == __main__.ppoint
            type(toPoint([[5,7],[13,4]])) == __main__.ppoint
        '''
        # For read-continuity between class and instance methods
        ppoint_out = cls(point_ppoint_listOfPoints_listOfPPoints)
        return ppoint_out
    
    @classmethod
    def random(cls, lowerlim, upperlim):
        x = random.randint(lowerlim, upperlim)
        y = random.randint(lowerlim, upperlim)
        return cls([x, y])
    
    @staticmethod
    def __distance__(s_ppoint, m_ppoint):
        return np.sqrt((m_ppoint.x-s_ppoint.x)**2+(m_ppoint.y-s_ppoint.y)**2)
    
    def distance(self, point_or_list):
        '''
        Returns the euclidean distance between the origin point and (an)other point(s).
        The origin point is the point on which the method is applied.
        Takes as input (a list of) coordinates or a (list of) ppoint(s)
        '''
        m_ppoint = self.toPoint(point_or_list) 
        return self.__distance__(self, m_ppoint)
    
    @staticmethod
    def __angleOffset__(s_ppoint, m_ppoint):
        atan2_v = np.vectorize(math.atan2)
        degrees_v = np.vectorize(math.degrees)  
        dx = m_ppoint.x - s_ppoint.x
        dy = m_ppoint.y - s_ppoint.y
        return degrees_v(atan2_v(dy, dx))
    
    def angleOffset(self, point_or_list):
        '''
        Returns the angle at which the horizontal line needs to rotate clockwise
        in order to match the line between the origin point and (an)other
        point(s).
        The origin point is the point on which the method is applied.
        Takes as input (a list of) coordinates or a (list of) ppoint(s)
        '''
        m_ppoint = self.toPoint(point_or_list) 
        return self.__angleOffset__(self, m_ppoint)
    
    @staticmethod
    def __centroid__(m_ppoint):
        n_points = len(m_ppoint)
        centroid = [m_ppoint.x.sum() / n_points, m_ppoint.y.sum() / n_points]
        return centroid
    
    @classmethod
    def centroid(cls, point_or_list):
        '''
        Returns the centroid of a (list of) point(s) as a ppoint
        Takes as input (a list of) coordinates or a (list of) ppoint(s)
        '''
        m_ppoint = cls.toPoint(point_or_list) 
        centroid = cls.__centroid__(m_ppoint)
        centroid_ppoint = cls.toPoint(centroid)
        return centroid_ppoint
    
    @classmethod
    def orderedIndex(cls, point_or_list):
        '''
        Returns the index of the points in clockwise order
        Takes as input (a list of) coordinates or a (list of) ppoint(s)
        '''
        centr = cls.centroid(point_or_list)
        #return points_list[centr.angleOffset(points_list).argsort()][::-1]
        return (centr.angleOffset(point_or_list)*-1).argsort(axis=0)

    @staticmethod
    def __polyArea__(m_ppoint):
        return 0.5*np.abs(np.dot(m_ppoint.x.T[0],np.roll(m_ppoint.y.T[0],1))
                          -np.dot(m_ppoint.y.T[0],np.roll(m_ppoint.x.T[0],1)))
        
    @classmethod
    def polyArea(cls, point_or_list):
        m_ppoint = cls.toPoint(point_or_list)
        m_ppoint = m_ppoint[cls.orderedIndex(point_or_list)][:,0]
        area = cls.__polyArea__(m_ppoint)
        return area