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
    
    
      ###############################
     #### Dunder and properties ####
    ###############################
    
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
    
    
      ######################################
     #### Random instantiation methods ####
    ######################################
    
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
    
    @classmethod
    def _populate_lines(cls, m_point, nr_circles, jitter_sd, radius_min, radius_max):
        '''
        Args:
            m_point: multiple Points
            nr_circles: number of circles to be created for each line
            jitter_sd: the standard deviation of the normal distribution
                from which the jitter is sampled
            radius_min: the minimum value for the radii
            radius_max: the maximum value for the radii
        Returns:
            A number of randomized circles between each set of Points in their 
            given order. This allows one to populate circles on the outer bound
            of irregular geometric shapes given by the corner Points.
        '''
        populated_lines = super(Circle, cls)._populate_lines(m_point, nr_circles, jitter_sd)
        random_radii = np.random.uniform(radius_min, radius_max, len(populated_lines))
        random_radii = random_radii.reshape(-1,1)
        populated_lines = np.append(populated_lines, random_radii, axis=1)
        return populated_lines
    
    @classmethod
    def populate_lines(cls, m_point, nr_circles, jitter_sd=1, radius_min=2, radius_max=2):
        '''
        Args:
            m_point: multiple Points
            nr_circles: number of circles to be created for each line
            jitter_sd: the standard deviation of the normal distribution
                from which the jitter is sampled
            radius_min: the minimum value for the radii
            radius_max: the maximum value for the radii
        Returns:
            A number of randomized Circles between each set of Points in their 
            given order. This allows one to populate Circles on the outer bound
            of irregular geometric shapes given by the corner Points.
        '''
        populated_lines = cls._populate_lines(m_point, nr_circles, jitter_sd, radius_min, radius_max)
        populated_lines_as_Circles = Circle(populated_lines)
        return populated_lines_as_Circles
    
    
      ######################
     #### Core methods ####
    ######################
    
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
        
        Returns: the xy-coordinates of the intersection points between s_circle 
            and m_circle as a numpy.ndarray
        '''
        # Rewriting parameters for conciseness
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
        
        Returns: the xy-coordinates of the intersection points between the
            origin Circle and the (multiple) circle(s) as a tuple of 2 Points
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
        return intersects

    def intersections(self):
        '''   
        Returns: the xy-coordinates of the intersection points between each of
            the origin Circles and the other origin Circles as a list of tulpes
            of 2 Points
        '''
        ###intersects = np.array([]).reshape(-1,2)
        intersects_l = list()

        for i in range(len(self)):
            intersects_st = self[i].intersect(self.drop(i))
            # Flatten the 2 Points by re-creating a Point.
            #intersects_st = super(Circle, self).__new__(Point, intersects_st)
            # Append to full list of intersects
            ###intersects = np.append(intersects, intersects_st, axis=0)
            intersects_l.append(intersects_st)
        return intersects_l ###super(Circle, self).__new__(Point, intersects)
    
    def encompass(self, s_point_or_circle, precision=8):
        '''
        Args:
            s_point_or_circle: a single Point or Circle to test
        
        Returns:
            Boolean indicating if the single Point or Circle is encompassed
        '''
        if isinstance(s_point_or_circle, Circle):
            s_circle = s_point_or_circle
            distances = self.distance(s_circle)
            lengths = distances + s_circle.r
            # Note the smaller than sign; Boundery values return False
            is_in_circle = (lengths < self.r).any() 
            return is_in_circle
        
        elif isinstance(s_point_or_circle, Point):
            s_point = s_point_or_circle
            r = self.r
            h = self.x
            k = self.y
            x = s_point.x
            y = s_point.y
            # Note the smaller than sign; Boundery values return False
            is_in_circle = ((x-h)**2+(y-k)**2).round(precision)<(r**2).round(precision)
            return is_in_circle
        
    def boundaries(self):
        '''
        Returns: the xy-coordinates of the intersection points between each of
            the origin Circles and the other origin Circles as a Point as well as
            the corresponding centerpoint. The result is returned only if the 
            aformentioned intersections are not encompassed by a Circle, i.e. 
            the intersections lies on the boundary of the Circles. The result
            is returned as a list of Points
        '''
        boundaries_l = []
        intersects_l = self.intersections()
        # Iterate through the intersections tulpes of each Circle
        for i, intersects_t in enumerate(intersects_l):
            # Each intersections tulpe exists of 2 Points (with each many entries)
            intersects_1, intersects_2 = intersects_t
            # We keep the other Circles which have intersections with Circle i
            nan_mask = np.array(np.isnan(intersects_1).any(axis=1))==False
            circles_without_i = self.drop(i)[nan_mask,:]
            # We drop empty intersections (resulting in a similar shape as circles_without_i)
            intersects_1, intersects_2 = intersects_1.dropna(), intersects_2.dropna()
            # For the non-empty intersections, we verify encompassment by the Circles
            outer_mask_1 = [self.encompass(intersect).any()==False for intersect in intersects_1]
            outer_mask_2 = [self.encompass(intersect).any()==False for intersect in intersects_2]
            # Store boundary values based on the encompassment masks
            boundaries_st = (Point([intersects_1[outer_mask_1],
                                    intersects_2[outer_mask_2]]),
                             Point([circles_without_i.xy[outer_mask_2],
                                    circles_without_i.xy[outer_mask_1]]),
                             self[i].xy)
            # Add to list if non-empty
            if len(boundaries_st[0])>0:
                boundaries_l.append(boundaries_st)
        return boundaries_l
    
    
      ##########################
     #### Obsolete methods ####
    ##########################
    
    @classmethod
    def _pointEncompassment(cls, s_point, m_circle):
        '''
        Args:
            m_point: Points which are tested for encompassment
            m_circle: Circles in which encompassment of the Points is tested
            
        Returns:
            Boolean for encompassment
        '''
        r = m_circle.r
        h = m_circle.x
        k = m_circle.y
        x = s_point.x
        y = s_point.y
        is_in_circle = ((x-h)**2+(y-k)**2)<(r**2)
        return is_in_circle
    
    def _encompassment(self, m_circle):
        '''
        Args:
            m_circle: The Circles in which encompassment is tested
            
        Returns:
            Boolean indicating at least one encompassment by a single Circle 
            from m_circle
        '''
        distances = self.distance(m_circle)
        lengths = distances + self.r
        flag_encompassment = (lengths < m_circle.r).any() 
        return flag_encompassment
    
    def overlap(self, m_circle):
        '''
        Args:
            m_circle: The Circles on which overlap is tested
            
        Returns:
            Boolean indicating full overlap by combining all Circles from m_circle
            Note that encompassment is an overlap by a single Circle
        '''
        return None
    
    
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