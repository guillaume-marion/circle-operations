# Library imports
import math
import numpy as np


# Defining the Point class
class Point(np.ndarray):
    '''
    The Point class is a child class of the numpy.ndarray. It includes a
        number of methods which can be used to calculate metrics between sets 
        of coordinates such as distances, etc.
    Note the difference in notation between:
        - 'point' which at least represent a set of coordinates
        - 'Point' which represents the class
    '''
    

      ###############################
     #### Dunder and properties ####
    ###############################
    
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
    def _reClass(cls, something_to_instantiate):
        '''
        Wraps the class instantiation. To be used for:
            - cleaner reading of code
            - allowing multiple input types in certain methods
        '''
        point_out = cls(something_to_instantiate)
        return point_out
    
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
        Typical usecase: selecting a single row from a multiple Point instance
            should also result in a Point with the correct shape to allow
            use of the class' methods.
        '''
        if type(val)==int:
            return self._reClass(np.asarray(self)[val])
        else:
            return super(Point, self).__getitem__(val)
    
    def __repr__(self):
        '''
        Defines custom __repr__ method.
        '''
        return (str(self.__class__.__name__)+"([\n {:>10}\n])"
                .format(self.__str__()[1:-1]))
    
    
      ######################################
     #### Random instantiation methods ####
    ######################################
    
    @staticmethod
    def _random(x_min, x_max, y_min, y_max, nr_points):
        '''
        Args:
            x_min: minium value for x-coordinates
            x_max: maximum value for x-coordinates
            y_min: minimum value for y-coordinates
            y_max: maximum value for y-coordinates
            nr_points: number of xy-coordinates to be produced
        
        Returns: 
            Random xy-coordinates
        '''
        x = np.random.uniform(x_min, x_max, nr_points)
        y = np.random.uniform(y_min, y_max, nr_points)
        xy = np.dstack([x,y])
        return xy
    
    @classmethod
    def random(cls, x_min, x_max, y_min, y_max, nr_points):
        '''
        Args:
            x_min: minium value for x-coordinates
            x_max: maximum value for x-coordinates
            y_min: minimum value for y-coordinates
            y_max: maximum value for y-coordinates
            nr_points: number of xy-coordinates to be produced
        
        Returns: 
            Random instance of a Point
        '''
        xy_values = cls._random(x_min, x_max, y_min, y_max, nr_points)
        random_point = cls(xy_values)
        return random_point
    
    @classmethod
    def _populate_lines(cls, m_point, nr_points, jitter_sd):
        '''
        Args:
            m_point: multiple Points
            nr_points: number of points to be created for each line
            jitter_sd: the standard deviation of the normal distribution
                from which the jitter is sampled
        Returns:
            A number of randomized points between each set of Points in their 
            given order. This allows one to populate points on the outer bound
            of irregular geometric shapes given by the corner Points.
        '''
        populated_lines = np.array([]).reshape(0,2)
        nr_segments = len(m_point)
        
        def _xRange(s_point, s_point_2):
            '''
            Args:
                s_point: A Point
                s_point_2: Another Point
            Returns:
                The minimum and maximum x-coordinates of 2 Points
            '''
            x_min = float(min(s_point.x, s_point_2.x))
            x_max = float(max(s_point.x, s_point_2.x))
            return x_min, x_max
    
        def _lineParameters(s_point, s_point_2):
            '''
            Args:
                s_point: A Point
                s_point_2: Another Point
            Returns:
                The coefficient and intercept of the line between 2 Points
            '''
            a = float((s_point_2.y-s_point.y)/(s_point_2.x-s_point.x))
            b = float(s_point.y - a*s_point.x)
            return a,b
    
        def _populate_line(s_point, s_point_2, nr_points, jitter_sd):
            '''
            Args:
                s_point: A Point
                s_point_2: Another Point
                nr_points: Number of points to produce
                jitter_sd: the standard deviation of the normal distribution
                    from which the jitter is sampled
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
    def populate_lines(cls, m_point, nr_points, jitter_sd=1):
        '''
        Args:
            m_point: multiple Points
            nr_points: number of Points to be created for each line
            jitter_sd: the standard deviation of the normal distribution
                from which the jitter is sampled
        Returns:
            A number of randomized Points between each set of Points in their 
            given order. This allows one to populate points on the outer bound
            of irregular geometric shapes given by the corner Points.
        '''
        populated_lines = cls._populate_lines(m_point, nr_points, jitter_sd)
        populated_lines_as_Points = Point(populated_lines)
        
        return populated_lines_as_Points

    
      ######################
     #### Core methods ####
    ######################

    def drop(self, row):
        '''
        Args:
            row: row-index to be dropped
        Returns:
            The Circle without the specified row
        '''
        lower_end = self[:row]
        upper_end = self[row+1:]
        appended = np.append(lower_end, upper_end, axis=0)
        appended_circle = self._reClass(appended)
        
        return appended_circle
    
    def dropna(self):
        '''
        Returns:
            Circle without nan values
        '''
        mask = np.array((np.isnan(self)==False).any(axis=1))
        self_without_nan = self[mask,:]
        
        return self_without_nan

    @staticmethod
    def _distance(s_point_x, s_point_y, m_point_x, m_point_y):
        '''
        Args:
            s_point_x: a single x-coordinate as an int, float or numpy.ndarray
            s_point_y: a single y-coordinate as an int, float or numpy.ndarray
            m_point_x: a single or multiple x-coordinates as a numpy.ndarray
            m_point_y: a single or multiple y-coordinates as a numpy.ndarray
        
        Returns: the euclidean distance(s) between the s_point and m_point as 
            a numpy.ndarray
        '''
        return np.sqrt((m_point_x-s_point_x)**2+(m_point_y-s_point_y)**2)
    
    def distance(self, point_or_list):
        '''
        Args:
            point_or_list: a list or numpy array of single/multiple 
                xy-coordinates or a (list of) Point(s).
        
        Returns: the euclidean distance(s) between the origin and the (multiple) 
            point(s) as a numpy.ndarray.
        '''
        s_point_x = self.x
        s_point_y = self.y
        m_point = self._reClass(point_or_list)
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
            s_point_x: a single x-coordinate as an int, float or numpy.ndarray
            s_point_y: a single y-coordinate as an int, float or numpy.ndarray
            m_point_x: a single or multiple sets of x-coordinates as a
                numpy.ndarray
            m_point_y: a single or multiple sets of y-coordinates as a
                numpy.ndarray
        
        Returns: the angle at which the horizontal line needs to rotate
            clockwise in order to match the line between the s_point and
            the m_point.
        '''
        atan2_v = np.vectorize(math.atan2)
        degrees_v = np.vectorize(math.degrees)  
        dx = m_point_x - s_point_x
        dy = m_point_y - s_point_y
        return degrees_v(atan2_v(dy, dx))
    
    def angleOffset(self, point_or_list):
        '''
        Args:
            point_or_list: a list or numpy array of single/multiple 
                xy-coordinates or a (list of) Point(s).
        
        Returns: the angles at which the horizontal line needs to rotate
            clockwise in order to match the line between the origin and
            the (multiple) point(s) as a numpy.ndarray.
        '''
        s_point_x = self.x
        s_point_y = self.y
        m_point = self._reClass(point_or_list)
        m_point_x = m_point.x
        m_point_y = m_point.y
        return self._angleOffset(s_point_x=s_point_x,
                                 s_point_y=s_point_y,
                                 m_point_x=m_point_x,
                                 m_point_y=m_point_y)
    
    @staticmethod
    def _centroid(m_point_x, m_point_y):
        '''
        Args:
            m_point_x: a single or multiple x-coordinates as a numpy.ndarray
            m_point_y: a single or multiple y-coordinates as a numpy.ndarray
        
        Returns: the xy-coordinates of the centroid as a numpy.ndarray
        '''
        n_points = len(m_point_x)
        centroid = [m_point_x.sum() / n_points, m_point_y.sum() / n_points]
        return centroid
    
    def centroid(self):
        '''       
        Returns: the xy-coordinates of the centroid of the Point instance as a Point
        '''
        m_point_x = self.x
        m_point_y = self.y
        centroid = self._centroid(m_point_x=m_point_x,
                                  m_point_y=m_point_y)
        centroid_point = Point(centroid)
        return centroid_point
    
    @staticmethod
    def _orderedIndex(some_angle_offsets):
        '''
        Args:
            some_angle_offsets: a result from the angleOffset method
        
        Returns: the index of the points in a clockwise order as a numpy.ndarray
        '''
        ordered_index = (some_angle_offsets*-1).argsort(axis=0)
        return ordered_index

    def orderedPoints(self):
        '''
        Returns:
            The Points in a clockwise-ordered fashion
        '''
        centr = self.centroid()
        angle_offsets = centr.angleOffset(self)
        ordered_index = self._orderedIndex(angle_offsets)
        ordered_self = self[ordered_index][:,0]
        
        return ordered_self

    @staticmethod
    def _polyArea(m_point_ordered_x, m_point_ordered_y):
        '''
        Args:
            m_point_ordered_x: multiple x-coordinates in clockwise order
                as a numpy.ndarray
            m_point_ordered_y: multiple y-coordinates in clockwise order
                as a numpy.ndarray
        
        Returns: the area of the polygon bounded by the points as a numpy.ndarray
        '''
        return 0.5*np.abs(np.dot(m_point_ordered_x.T[0],np.roll(m_point_ordered_y.T[0],1))
                          -np.dot(m_point_ordered_y.T[0],np.roll(m_point_ordered_x.T[0],1)))

    def polyArea(self):
        '''
        Returns: the area of the polygon bounded by the Point instance as a numpy.ndarray
        '''
        m_point = self.xy
        m_point_ordered = m_point.orderedPoints()
        m_point_ordered_x = m_point_ordered.x
        m_point_ordered_y = m_point_ordered.y
        area = self._polyArea(m_point_ordered_x=m_point_ordered_x,
                              m_point_ordered_y=m_point_ordered_y)
        return area
