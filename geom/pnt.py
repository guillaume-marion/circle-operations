# Library imports
import math
import numpy as np





# Defining the Point class
class Point(np.ndarray):
    '''
    The Point class is a child class of the numpy.ndarray. It includes a
        number of methods which can be used to calculate metrics between sets 
        of coordinates such as distances, angles, ...
    Note the difference in notation between (not applicable in arguments of methods):
        - 'point' which at least represent a set of coordinates in a list, np.ndarray,...
        - 'Point' which represents an instance of the class.
    Also note that when we refer to Points we mean a single instance of the object
        existing of multiple rows.
    '''

      ###############################
     #### Dunder and properties ####
    ###############################
    
    def __new__(cls, inputarray):
        '''
        A Point is created from a np.ndarray.
        The array can exist of a single or multiple points, e.g.:
        - a = Point([5,7])
        - b = Point([[5,7],[13,4]])  
        '''
        obj = np.asarray(inputarray).view(cls)
        try:
            # Controlling for correct shape
            obj - np.asarray([1, 1])
            # Reshaping to columns
            obj = obj.reshape(-1, 2)
            return obj
        except:
            raise ValueError("The input should have the shape of a (2,) or (*,2) array.") 
    
    def __init__(self, inputarray):
        '''
        We use the return from __new__ as self.
        '''
        pass
    
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
    @property
    def xy(self):
        return self[:,:]
    @xy.setter
    def xy(self, value):
        self[:,:] = value
     
        
    def __getitem__(self,val):
        '''
        If val is an int, then consider it as a row-selection which should
            result in a Point. In every other case we use numpy's  __getitem__ method.
        Usecase: Selecting a single row from a multiple Point instance
            should also result in a Point with the correct shape (i.e. double 
            enclosing brackets) to allow further use of the class' methods.
        '''
        if type(val)==int:
            return self.__class__(np.asarray(self)[val])
        else:
            return super(Point, self).__getitem__(val)
    
    def __repr__(self):
        '''
        Defines custom __repr__ method.
        '''
        return (str(self.__class__.__name__)+"([\n {:>10}\n])"
                .format(self.__str__()[1:-1]))
        
    def __round__(self, decimals=0):
        '''
        Defines custome __round__ method, which we can use to return a class
            compared to an array with .round().
        '''
        rounded_array = self.round(decimals)
        rounded_class = self.__class__(rounded_array)
        return rounded_class
    
      #######################################
     #### Random initialization methods ####
    #######################################
    
    @staticmethod
    def _random(x_min, x_max, y_min, y_max, nr_points):
        '''
        Args:
            x_min: Minium value for x-coordinates.
            x_max: Maximum value for x-coordinates.
            y_min: Minimum value for y-coordinates.
            y_max: Maximum value for y-coordinates.
            nr_points: The number of points to be produced.
        
        Returns: 
            A np.ndarray of random points.
        '''
        x = np.random.uniform(x_min, x_max, nr_points)
        y = np.random.uniform(y_min, y_max, nr_points)
        xy = np.dstack([x,y])
        return xy
    
    @classmethod
    def random(cls, x_min, x_max, y_min, y_max, nr_points):
        '''
        Args:
            x_min: Minium value for x-coordinates.
            x_max: Maximum value for x-coordinates.
            y_min: Minimum value for y-coordinates.
            y_max: Maximum value for y-coordinates.
            nr_points: The number of Points to be produced.
        
        Returns: 
            A random instance of (a) Point(s).
        '''
        xy_values = cls._random(x_min, x_max, y_min, y_max, nr_points)
        random_point = cls(xy_values)
        return random_point
    
    
      ######################
     #### Core methods ####
    ######################

    def drop(self, row):
        '''
        Args:
            row: The index of the row to be dropped.
            
        Returns:
            The Point without the specified row.
        '''
        lower_end = self[:row]
        upper_end = self[row+1:]
        appended = np.append(lower_end, upper_end, axis=0)
        appended_point = self.__class__(appended)
        return appended_point
    
    def dropna(self):
        '''
        Returns:
            The Point without nan values.
        '''
        mask = np.array((np.isnan(self)==False).any(axis=1))
        self_without_nan = self[mask,:]
        return self_without_nan

    @staticmethod
    def _distance(s_point_x, s_point_y, m_point_x, m_point_y):
        '''
        Args:
            s_point_x: A single x-coordinate as an int, float or numpy.ndarray.
            s_point_y: A single y-coordinate as an int, float or numpy.ndarray.
            m_point_x: A single or multiple x-coordinates as a numpy.ndarray.
            m_point_y: A single or multiple y-coordinates as a numpy.ndarray.
        
        Returns: The euclidean distance(s) between the s_point and m_point as 
            a numpy.ndarray.
        '''
        return np.sqrt((m_point_x-s_point_x)**2+(m_point_y-s_point_y)**2)
    
    def distance(self, point):
        '''
        Args:
            point: One or multiple Points.
        
        Returns: The euclidean distance(s),  as a numpy.ndarray, between self 
            and the (multiple) Point(s).
        '''
        s_point_x = self.x
        s_point_y = self.y
        m_point = point
        m_point_x = m_point.x
        m_point_y = m_point.y
        return self._distance(s_point_x=s_point_x,
                              s_point_y=s_point_y,
                              m_point_x=m_point_x,
                              m_point_y=m_point_y)
    
    @staticmethod
    def _angleOffset(s_point_x, s_point_y, m_point_x, m_point_y):
        '''
        Args:
            s_point_x: A single x-coordinate as an int, float or numpy.ndarray.
            s_point_y: A single y-coordinate as an int, float or numpy.ndarray.
            m_point_x: A single or multiple x-coordinates as a numpy.ndarray.
            m_point_y: A single or multiple y-coordinates as a numpy.ndarray.
        
        Returns: The angle at which the horizontal line needs to rotate
            clockwise in order to match the line between the s_point and
            the m_point.
        '''
        atan2_v = np.vectorize(math.atan2)
        degrees_v = np.vectorize(math.degrees)  
        dx = m_point_x - s_point_x
        dy = m_point_y - s_point_y
        return degrees_v(atan2_v(dy, dx))
    
    def angleOffset(self, point):
        '''
        Args:
            point: One or multiple Points.
        
        Returns: The angles, as a numpy.ndarray, at which the horizontal line 
            needs to rotate clockwise in order to match the line between self and
            the (multiple) Point(s).
        '''
        s_point_x = self.x
        s_point_y = self.y
        m_point = point
        m_point_x = m_point.x
        m_point_y = m_point.y
        return self._angleOffset(s_point_x=s_point_x,
                                 s_point_y=s_point_y,
                                 m_point_x=m_point_x,
                                 m_point_y=m_point_y)
    
    def angleBetween(self, point, starting_point):
        '''
        Args:
            point: One or multiple Points.
            starting_point: A Point from which the clockwise angle to the other
                Points is calculated.
                
        Returns: The angles, as a np.ndarray, between the starting_point and the 
            Point(s) with self as vertex.
        '''
        angleOrigin = 180-self.angleOffset(starting_point)
        anglePoints = 180-self.angleOffset(point)
        anglePoints[anglePoints<angleOrigin] = anglePoints[anglePoints<angleOrigin]+360
        return anglePoints-angleOrigin
    
    @staticmethod
    def _centroid(m_point_x, m_point_y):
        '''
        Args:
            m_point_x: Multiple x-coordinates as a numpy.ndarray.
            m_point_y: Multiple y-coordinates as a numpy.ndarray.
        
        Returns: The point of the centroid as a numpy.ndarray.
        '''
        n_points = len(m_point_x)
        centroid = [m_point_x.sum() / n_points, m_point_y.sum() / n_points]
        return centroid
    
    def centroid(self):
        '''       
        Returns: The centroid of the Points (i.e. self) as a single Point.
        '''
        m_point_x = self.x
        m_point_y = self.y
        centroid = self._centroid(m_point_x=m_point_x,
                                  m_point_y=m_point_y)
        centroid_point = Point(centroid)
        return centroid_point
    
    @staticmethod
    def _orderedIndex(some_angles):
        '''
        Args:
            some_angles: A least two angles in a np.ndarray.
        
        Returns: The index, as a numpy.ndarray, of the angles in a clockwise order.
        '''
        ordered_index = (some_angles).argsort(axis=0)
        return ordered_index

    def orderedPoints(self, centerpoint=None, start_from=None, return_angles=False):
        '''
        Args:
            centerpoint: Defaults to the centroid of self as the centerpoint for 
                the clock-pointer. Otherwise a given center Point is used.
            start_from: Defaults to the horizontal line as starting point. 
                Else if given a Point, it will serve as the starting point 
                for computing the clockwise order.
                
        Returns:
            The Points in a clockwise-ordered fashion.
        '''
        if centerpoint is None:
            centr = self.centroid()
        else:
            centr = centerpoint
            
        if start_from is None:
            angles = centr.angleOffset(self)*-1
        else:
            angles = centr.angleBetween(self, start_from)
            
        ordered_index = self._orderedIndex(angles)
        ordered_self = self[ordered_index][:,0]
        if return_angles:
            angles.sort()
            return ordered_self, angles
        else:
            return ordered_self

    @staticmethod
    def _polyArea(m_point_ordered_x, m_point_ordered_y):
        '''
        Args:
            m_point_ordered_x: Multiple x-coordinates in clockwise order
                as a numpy.ndarray.
            m_point_ordered_y: Multiple y-coordinates in clockwise order
                as a numpy.ndarray.
        
        Returns: The area, as a numpy.ndarray, of the polygon bounded by the points.
        '''
        return 0.5*np.abs(np.dot(m_point_ordered_x.T[0],np.roll(m_point_ordered_y.T[0],1))
                          -np.dot(m_point_ordered_y.T[0],np.roll(m_point_ordered_x.T[0],1)))
    
    def polygonArea(self):
        '''
        Returns:
            The area, as a numpy.ndarray, of the polygon bounded by the Points .
        '''
        m_point_ordered_x = self.x
        m_point_ordered_y = self.y 
        A = self._polyArea(m_point_ordered_x, m_point_ordered_y)
        return A
    
    def polyEncompass(self, point, func=all):
        '''
        Args:
            point: The point(s) which needs to be verified if encompassed by the
                polygon desfined by self.
            func: The aggregation function for the boolean list.
                
        Returns:
            A boolean indicating encompassment of the Point by the polygon.
        '''
        # Relabel point for conciseness.
        mp = point
        # Define the segments as a list of Points of 2 points.
        segments = [self[[i,i+1]] for i in range(len(self)-1)]
        # Receptacle for all ecompassment tests.
        all_encompassments = []

        for p in mp:
            # Number of crossing numbers.
            nr_cn = 0 
            # For each segment...
            for s0,s1 in segments:
                # If there is an upward ..or.. downward crossing.
                if ((s0.y <= p.y and s1.y > p.y)   
                    or (s0.y > p.y and s1.y <= p.y)):
                    # We compute the edge-ray intersect x-coordinate.
                    vt = (p.y - s0.y) / float(s1.y - s0.y)
                    if p.x < s0.x + vt * (s1.x - s0.x):
                        nr_cn += 1  
            overlap = np.array((p==self).all(axis=1)).any()
            encompassment = bool(nr_cn % 2) or overlap
            all_encompassments.append(encompassment)
    
        return func(all_encompassments)