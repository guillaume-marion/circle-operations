# Library imports
import math
import numpy as np





# Defining the Point class
class Point(np.ndarray):
    '''
    The Point class is a child class of the numpy.ndarray. It includes a
        number of methods which can be used to calculate metrics between sets 
        of coordinates such as distances, angles, ...
    Note the difference in notation between:
        - 'point' which at least represent a set of coordinates in a list, array,...
        - 'Point' which represents an instance of the class.
    '''

      ###############################
     #### Dunder and properties ####
    ###############################
    
    def __new__(cls, inputarray):
        '''
        A Point is created from a np.ndarray.
        The array can exist of a single set of coordinates or multiple, e.g.:
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
            result in a Point instance. If a slice, then use numpy's 
            __getitem__ method.
        Typical usecase: Selecting a single row from a multiple Point instance
            should also result in a Point with the correct shape to allow
            use of the class' methods.
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
            nr_points: The number of xy-coordinates to be produced.
        
        Returns: 
            A np.ndarray of random xy-coordinates.
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
            nr_points: The number of xy-coordinates to be produced.
        
        Returns: 
            A random instance of a Point.
        '''
        xy_values = cls._random(x_min, x_max, y_min, y_max, nr_points)
        random_point = cls(xy_values)
        return random_point
    
    @classmethod
    def _populate_lines(cls, m_point, nr_points, jitter_sd):
        '''
        Args:
            m_point: Multiple Points.
            nr_points: The number of points to be created.
            jitter_sd: The standard deviation of the normal distribution
                from which the the jitter is sampled.
                
        Returns:
            A number of randomized points between each set of Points in their 
            given order. This allows one to populate points along a path defined
            by a set of given Points.
        '''
        populated_lines = np.array([]).reshape(0,2)
        nr_segments = len(m_point)
        
        def _xRange(s_point, s_point_2):
            '''
            Args:
                s_point: A Point.
                s_point_2: Another Point.
            Returns:
                The minimum and maximum values of the x-coordinates of the 2 Points.
            '''
            x_min = float(min(s_point.x, s_point_2.x))
            x_max = float(max(s_point.x, s_point_2.x))
            return x_min, x_max
    
        def _lineParameters(s_point, s_point_2):
            '''
            Args:
                s_point: A Point.
                s_point_2: Another Point.
            Returns:
                The coefficient and intercept of the line between the 2 Points.
            '''
            a = float((s_point_2.y-s_point.y)/(s_point_2.x-s_point.x))
            b = float(s_point.y - a*s_point.x)
            return a,b
    
        def _populate_line(s_point, s_point_2, nr_points, jitter_sd):
            '''
            Args:
                s_point: A Point.
                s_point_2: Another Point.
                nr_points: The number of points to produce.
                jitter_sd: The standard deviation of the normal distribution
                    from which the jitter is sampled.
            Returns:
                Randomized points on the fitted line between 2 Points taking into
                consideration the maximum and minimum x-coordinates of these 2
                Points. Also some jitter (based on a normal distribution) is
                added as vertical distance from the line.
            '''
            a,b = _lineParameters(s_point, s_point_2)
            x_min, x_max = _xRange(s_point, s_point_2)
            x = np.random.uniform(x_min, x_max, nr_points)
            jitter_values = np.random.normal(0, jitter_sd, nr_points)
            y = (a*x) + b + jitter_values
            return np.array(list(zip(x,y)))
    
        for i in range(nr_segments-1):
            populated_line = _populate_line(m_point[i], m_point[i+1], nr_points, jitter_sd)
            populated_lines = np.append(populated_line, populated_lines, axis=0)
        
        return populated_lines
    
    @classmethod
    def populate_lines(cls, point, nr_points, jitter_sd=1):
        '''
        Args:
            point: Multiple Points.
            nr_points: The number of Points to be created for each line.
            jitter_sd: the standard deviation of the normal distribution
                from which the jitter is sampled.
        Returns:
            A number of randomized Points between each set of Points in their 
            given order. This allows one to populate Points along a path defined
            by a set of given Points.
        '''
        populated_lines = cls._populate_lines(point, nr_points, jitter_sd)
        populated_lines_as_Points = Point(populated_lines)
        
        return populated_lines_as_Points

    
      ######################
     #### Core methods ####
    ######################

    def drop(self, row):
        '''
        Args:
            row: Row-index to be dropped.
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
        
        Returns: The euclidean distance(s) between the origin and the (multiple) 
            Point(s) as a numpy.ndarray.
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
            m_point_x: A single or multiple sets of x-coordinates as a
                numpy.ndarray.
            m_point_y: A single or multiple sets of y-coordinates as a
                numpy.ndarray.
        
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
        
        Returns: The angles at which the horizontal line needs to rotate
            clockwise in order to match the line between the origin and
            the (multiple) point(s) as a numpy.ndarray.
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
            point_or_list: One or multiple Points.
            starting_point: A Point from which the clockwise angle to the other
                Points is calculated.
                
        Returns: The angles between the starting_point and the point_or_list
            with the origin as vertex.
        '''
        angleOrigin = 180-self.angleOffset(starting_point)
        anglePoints = 180-self.angleOffset(point)
        anglePoints[anglePoints<angleOrigin] = anglePoints[anglePoints<angleOrigin]+360
        return anglePoints-angleOrigin
    
    @staticmethod
    def _centroid(m_point_x, m_point_y):
        '''
        Args:
            m_point_x: A single or multiple x-coordinates as a numpy.ndarray.
            m_point_y: A single or multiple y-coordinates as a numpy.ndarray.
        
        Returns: The xy-coordinates of the centroid as a numpy.ndarray.
        '''
        n_points = len(m_point_x)
        centroid = [m_point_x.sum() / n_points, m_point_y.sum() / n_points]
        return centroid
    
    def centroid(self):
        '''       
        Returns: The xy-coordinates of the centroid of the Point instance as a Point.
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
            some_angle_offsets: A number of angles to sort.
        
        Returns: The index of the points in a clockwise order as a numpy.ndarray.
        '''
        ordered_index = (some_angles).argsort(axis=0)
        return ordered_index

    def orderedPoints(self, centerpoint=None, start_from=None, return_angles=False):
        '''
        Args:
            centerpoint: Either the centroid as the origin for the clock-pointer 
                or a given center Point.
            start_from: If given a Point, it will serve as the starting point 
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
        
        Returns: The area of the polygon bounded by the points as a numpy.ndarray.
        '''
        return 0.5*np.abs(np.dot(m_point_ordered_x.T[0],np.roll(m_point_ordered_y.T[0],1))
                          -np.dot(m_point_ordered_y.T[0],np.roll(m_point_ordered_x.T[0],1)))
    
    def polygonArea(self):
        '''
        Returns:
            The area of the polygon bounded by the points as a numpy.ndarray.
        '''
        m_point_ordered_x = self.x
        m_point_ordered_y = self.y 
        A = self._polyArea(m_point_ordered_x, m_point_ordered_y)
        return A