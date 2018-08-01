# Local imports
from geom.pnt import Point

# Library imports
import math
import warnings
import numpy as np





class Circle(Point):
    '''
    The Circle class is a child class of the Point. It includes a
        number of methods which can be used to calculate metrics between sets 
        of coordinates+radius such as intersections, ....
    Note the difference in notation between (not applicable in arguments of methods):
        - 'circle' which at least represent a set of coordinates+radius in a list, array, ...
        - 'Circle' which represents an instance of the class.
    Also note that when we refer to Circles we mean a single instance of the object
        existing of multiple rows.
    '''
    
      ###############################
     #### Dunder and properties ####
    ###############################
    
    def __new__(cls, inputarray):
        '''
        A Circle is created from a np.ndarray.
        The array can exist of a single or multiple circles, e.g.:
        - Circle([5,7,1])
        - Circle([[5,7,1],[13,4,1]])  
        '''
        obj = np.asarray(inputarray).view(cls)
        try:
            # Controlling for correct shape
            obj - np.asarray([1, 1, 1])
            # Reshaping to columns
            obj = obj.reshape(-1, 3)
            return obj
        except:
            raise ValueError("The input should have the shape of a (3,) or (*,3) array") 
    
    def __init__(self, inputarray):
        '''
        We add default parameters to control for clustering, etc.
        We use the return from __new__ as self.
        '''
        self.intersections = []
        self._clusters_indices = []
        self.nr_clusters = None
        self.isclustered = False
        self.iscluster = False
        self.isbounded = False
        self.outer_boundaries = []
        self.inner_boundaries = []
    
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
    
      #######################################
     #### Random initialization methods ####
    #######################################
    
    @classmethod
    def _random(cls, x_min, x_max, y_min, y_max, radius_min, radius_max, nr_circles):
        '''
        Args:
            x_min: Minium value for x-coordinates.
            x_max: Maximum value for x-coordinates.
            y_min: Minimum value for y-coordinates.
            y_max: Maximum value for y-coordinates.
            radius_min: Minimum value for radii.
            radius_max: Maximum value for radii.
            nr_circles: The number of circles to be produced.
        
        Returns: 
            A np.ndarray of random circles.
        '''
        xy = super(Circle, cls)._random(x_min, x_max, y_min, y_max, nr_circles)
        r = np.random.uniform(radius_min, radius_max, nr_circles)
        xyr = np.dstack([xy,r])
        return xyr
    
    @classmethod
    def random(cls, x_min, x_max, y_min, y_max, radius_min, radius_max, nr_circles):
        '''
        Args:
            x_min: Minium value for x-coordinates.
            x_max: Maximum value for x-coordinates.
            y_min: Minimum value for y-coordinates.
            y_max: Maximum value for y-coordinates.
            radius_min: Minimum value for radii.
            radius_max: Maximum value for radii.
            nr_circles: The number of Circles to be produced.
        
        Returns: 
            A random instance of (a) Circle(s).
        '''
        xyr_values = cls._random(x_min, x_max, y_min, y_max, radius_min, radius_max, nr_circles)
        random_circle = cls(xyr_values)
        return random_circle
    
    @classmethod
    def _populate_lines(cls, m_point, nr_circles, jitter_sd, radius_min, radius_max):
        '''
        Args:
            m_point: At least two Points.
            nr_circles: The number of circles to be created for each line.
            jitter_sd: The standard deviation of the normal distribution
                from which the jitter is sampled.
            radius_min: The minimum value for the radii.
            radius_max: The maximum value for the radii.
            
        Returns:
            A number of randomized circles between each set of Points in their 
            given order. This allows one to populate circles along a path defined
            by a set of given Points.
        '''
        # Populate points
        populated_lines = super(Circle, cls)._populate_lines(m_point, nr_circles, jitter_sd)
        # Add random radii
        random_radii = np.random.uniform(radius_min, radius_max, len(populated_lines))
        random_radii = random_radii.reshape(-1,1)
        populated_lines = np.append(populated_lines, random_radii, axis=1)
        return populated_lines
    
    @classmethod
    def populate_lines(cls, point, nr_circles, jitter_sd=1, radius_min=2, radius_max=2):
        '''
        Args:
            m_point: At least two Points.
            nr_circles: The number of Circles to be created for each line.
            jitter_sd: The standard deviation of the normal distribution
                from which the jitter is sampled.
            radius_min: The minimum value for the radii.
            radius_max: The maximum value for the radii.
        Returns:
            A number of randomized Circles between each set of Points in their 
            given order. This allows one to populate Circles along a path defined
            by a set of given Points.
        '''
        # Produce circles with the given args.
        populated_lines = cls._populate_lines(point, nr_circles, jitter_sd, radius_min, radius_max)
        # Initialize the circles as Circles.
        populated_lines_as_Circles = Circle(populated_lines)
        
        return populated_lines_as_Circles
    
    
      ######################
     #### Core methods ####
    ######################
    
    def area(self):
        '''
        Returns: The area(s) of the Circle(s) as a numpy.ndarray.
        '''
        return math.pi*self.r**2
    
    @staticmethod
    def _intersect(s_circle_r, s_circle_xy, m_circle_r, m_circle_xy, distance):
        '''
        Args:
            s_circle_r: A single radius as a numpy.ndarray.
            s_circle_xy: A single set of xy-coordinates as a numpy.ndarray.
            m_circle_r: A single or multiple radii as a numpy.ndarray.
            m_circle_xy: A single or multiple xy-coordinates as a numpy.ndarray.
            distance: The distance(s) between the s_circle and the m_circle(s).
        
        Returns: The intersection points between s_circle  and m_circle(s) as a 
            numpy.ndarray.
        '''
        # Rewriting parameters for conciseness
        d = distance
        r0 = s_circle_r
        r1 = m_circle_r
        xy0 = s_circle_xy
        xy1 = m_circle_xy
        # Raise if circles overlap.
        inf_intersects = ((d==0) & (r0==r1))
        if inf_intersects.sum()>0:
            raise OverflowError('tangent circles')
        # Warn for non-intersecting circles.
        mask = ((d>r0+r1) | (d<abs(r0-r1)))
        if mask.sum()>0:
            warnings.warn('no intersection')
        # Compute intersections in a vectorized fashion.
        ### Comppute the distance between self and the line passing the intersection 
        ###  points.
        a = (r0**2-r1**2+d**2) / (2*d)
        ### Set distance to nan if the is no intersections (this allows proper
        ###  execution of the code, and resulting in a tuple of np.nan values)
        a[mask]=np.nan
        ### Compute the distance between the centerpoint of the intersection line 
        ###  and one intersection point.
        h = np.sqrt(r0**2-a**2)
        ### First half of the sum.
        summand_1 = xy0+a*(xy1-xy0)/d
        diff = xy1-xy0
        ### Second half of the sum.
        summand_2 = (h*(diff)/d)[:,::-1]
        ### Numpy manipulation for correct summation.
        intersect_1 = summand_1+summand_2*np.array([[-1,1]])
        intersect_2 = summand_1+summand_2*np.array([[1,-1]])
        return intersect_1, intersect_2
    
    def intersect(self, circle):
        '''
        Args:
            circle: One or multiple Circles.
        
        Returns: The intersection points between self and the (multiple) 
            Circle(s) as a tuple of 2 Points (with each Point possibly holding 
            multiple points).
        '''
        # Extract the necessary parameters.
        s_circle_r = self.r
        s_circle_xy = self.xy
        m_circle = circle
        m_circle_r = m_circle.r
        m_circle_xy = m_circle.xy
        distance = self.distance(circle)
        # Execute the static method with above parameters.
        intersects = self._intersect(s_circle_r=s_circle_r,
                                     s_circle_xy=s_circle_xy,
                                     m_circle_r=m_circle_r,
                                     m_circle_xy=m_circle_xy,
                                     distance=distance)
        return intersects

    def calc_intersections(self):
        '''   
        Stores: 
            .intersections: The xy-coordinates of the intersection points between 
                each of self and the other origin Circles as a list of tulpes 
                of 2 Points (with each Point possibly holding multiple points).
        '''
        # Create a receptacle list.
        intersects_l = list()

        for i in range(len(self)):
            # Compute all the intersections between self and all other Circles.
            intersects_st = self[i].intersect(self.drop(i))
            # Append the intersects to the receptacle list.
            intersects_l.append(intersects_st)
            # Store results.
            #  We now have a list where the item with index i is a tuple of two
            #  Points. The first Point holds all first intersections with all the
            #  other Circles; The other holds all second intersections.
            self.intersections = intersects_l
    
    def encompass(self, s_point_or_circle, prec=8):
        '''
        Args:
            s_point_or_circle: A single Point or Circle to test.
        
        Returns:
            Boolean indicating if the single Point or Circle is encompassed.
        '''
        # Test if type is a Circle.
        if isinstance(s_point_or_circle, Circle):
            s_circle = s_point_or_circle
            distances = self.distance(s_circle)
            lengths = distances + s_circle.r
            # Note the smaller than sign; Boundary values will thus return False.
            is_in_circle = (lengths < self.r).any() 
            return is_in_circle
        
        # Test if type is a Point.
        elif isinstance(s_point_or_circle, Point):
            s_point = s_point_or_circle
            r = self.r
            h = self.x
            k = self.y
            x = s_point.x
            y = s_point.y
            # Note the smaller than sign; Boundary values will thus return False.
            is_in_circle = ((x-h)**2+(y-k)**2).round(prec)<(r**2).round(prec)
            return is_in_circle
    
    def calc_clusters(self):
        '''
        Stores: 
            .clusters_indices: A tuple of which each element is a set of indices 
                of an intersecting group of Circles.
            .nr_clusters: The number of identified clusters.
            .isclustered: Boolean indicating that the method has been executed.
        '''
        # Retrieve the intersections stored in self (for readability).
        intersections = self.intersections
        # Test that the intersections have been computed.
        if len(intersections) == 0:
            raise(RuntimeError, "To compute clusters, compute intersections first.")
        # Define constants
        nr_circles = len(self)
        index = np.arange(nr_circles)
        # Create receptacles.
        intersections_indices = []
        nr_intersections = []
        # For each Circle...
        for i,_ in enumerate(self.intersections):
            # ...create a mask of all the Circles which do not intersect with it.
            mask = np.isnan(_[0]).all(axis=1)==False
            # ...then get the indices of the Circle that DO intersect by using that mask.
            indices = set(np.delete(index, i)[mask])
            # ...also keep track of how many intersecting Circles that amounts to.
            nr_intersections.append(len(indices))
            # ...finally add the index of the Circle under scrutiny to the list
            #     of indices from above.
            indices.add(i)
            # ...and keep it in a list as a set.
            intersections_indices.append(indices)
        # Provide a starting point for finding the Circle-clusters by simply
        #  taking the first set of indices of intersecting Circles.
        # Note that a 'cluster' is defined simply as a set of Circle-indices.
        first_cluster_contribution = intersections_indices[0]
        cluster_list =  [first_cluster_contribution]
        # For each cluster (i.e. set of indices of Circles) in our list of clusters:
        for i, cluster in enumerate(cluster_list):
            # We make a first proposal for the final cluster definition, from which
            #  we will later built onto (i.e. possibly add more indices that belong
            #  to that cluster).
            final_cluster = cluster
            # We define the remaining indices as those indices that we need to
            #  evaluate for possible addition to our cluster definition.
            #  Namely, the list of sets of indices from Circles that intersect
            #  with eachother.
            remaining_indices = intersections_indices.copy()
            # We loop indefinitely...
            while True:
                added = False
                # For the further expansion of the definition of the cluster 
                #  under consideration we iterate over each set of indices 
                #  that intersect with each other.
                for indices in remaining_indices:
                    # If a Circle's index or the index of one of its
                    #  intersections is found in the cluster, then we add those indices
                    #  to the cluster.
                    if len(final_cluster & indices)>0:
                        final_cluster = final_cluster | indices
                        remaining_indices.remove(indices)
                        added = True
                # ... until no new addition to our current cluster is made.
                if not added:
                    cluster_list[i] = final_cluster
                    break
            # For the loop to continue we need a proposal for a next cluster.
            # We create a said proposal with an indices-group (i.e. a Circle's index
            #  as well as its intersection's indices) of which none of the indices 
            #  can be found in any of the previously constructed clusters.
            all_indices = set([item for sublist in cluster_list for item in sublist])
            new_clusters_candidates = ([sett for sett in remaining_indices 
                                        if len(sett & all_indices)==0 and len(sett)>0])          
            if len(new_clusters_candidates)>0:
                cluster_list.append(new_clusters_candidates[0])
            # We stop the loop if no new cluster (proposal) is created.
            else:
                break
        # We define some staging results: all clusters, clusters with multiple 
        #  Circles and clusters with only one Circle.
        all_clusters = cluster_list
        cluster_groups = [_ for _ in all_clusters if len(_)>1]
        cluster_isolats = [_ for _ in all_clusters if len(_)==1]
        # We do not include encompassed Circles in the clusters definition, as
        #  these Circles do not add any information to area calculations.
        for isolat in cluster_isolats:
            encompassment = self.encompass(self[list(isolat)])
            if encompassment == False:
                cluster_groups.append(isolat)
        # Store results.
        self._clusters_indices = cluster_groups
        self.nr_clusters = len(cluster_groups)
        self.isclustered = True
    
    def get_cluster(self, cluster_index):
        '''
        Args:
            cluster_index: The index of the cluster to retrieve.
            
        Returns:
            A new Circle instance existing of all the Circles pertaining to a
            given cluster.
        '''
        
        def _intersect_indices(indices, self_index):
            '''
            Args:
                indices: The indices of the given Circles within the cluster.
                self_index: The index of the Circle (self) within the cluster.
            
            Returns:
                A list of (shifted) indices that can be used to slice the 
                .intersections property to get those intersections between the Circle 
                (with cluster-index = self_index) and the other Circles 
                (with cluster-indices = indices).
                
            Note: This function is needed since the .intersect() method results in n-1
                intersections in the event of n Circles; Therefore resulting in an
                index one element shorted than the index of the original Circles.
            '''
            corrected_indices = []
            lower_than = [_ for _ in indices if _<self_index]
            corrected_indices.extend(lower_than)
            higher_than = [_-1 for _ in indices if _>self_index]
            corrected_indices.extend(higher_than)
            return corrected_indices
        
        # Test that clusters have been computed.
        if self.isclustered == False:
            raise(RuntimeError, "To extract a cluster, compute clusters first.")
        # Get the specific set of indices as a list and sort.
        indices_list = list(self._clusters_indices[cluster_index])
        indices_list.sort()
        # Create a new Circle from the relevant portion of self.
        Clustered_circle = Circle(self[indices_list])
        # Get the intersections Points for the Circles in the cluster.
        #  Note that we have to rework the Points to exclude all intersections
        #  between the Cluster's Circles and Circles not included in the cluster.
        #  (We could just recompute the .intersections() method on the new instance.)
        all_intersections = [self.intersections[_] for _ in indices_list]
        lean_intersections = []
        for i,index in enumerate(indices_list):
            i1,i2 = all_intersections[i]
            int_indices = _intersect_indices(indices_list, index)
            lean_intersections.append((i1[int_indices],i2[int_indices]))
        # Redefine variables
        Clustered_circle.intersections = lean_intersections
        Clustered_circle.iscluster = True
        Clustered_circle.isclustered = True # Needed because new instance...
        Clustered_circle.nr_clusters = 1
        return Clustered_circle
        
    
    def _all_boundaries(self):
        '''
        Returns: The intersection Points between each of the origin Circles and 
            the other origin Circles as a Point as well as the corresponding Circles 
            if the aformentioned intersections are not encompassed by a Circle, 
            i.e. the intersections lies on the boundary of the Circles.
        '''
        boundaries_l = []
        circles_l = []
        intersects_l = self.intersections
        # Iterate through the intersections tulpes of each Circle
        for i, intersects_t in enumerate(intersects_l):
            # Each intersections tulpe exists of 2 Points (with each many entries)
            intersects_1, intersects_2 = intersects_t
            # We drop empty intersections (resulting in a similar shape as circles_without_i)
            intersects_1, intersects_2 = intersects_1.dropna(), intersects_2.dropna()
            # For the non-empty intersections, we verify encompassment by the Circles
            outer_mask_1 = [self.encompass(intersect).any()==False for intersect in intersects_1]
            outer_mask_2 = [self.encompass(intersect).any()==False for intersect in intersects_2]
            # Store boundary values based on the encompassment masks
            boundaries_st = (Point([intersects_1[outer_mask_1],
                                    intersects_2[outer_mask_2]]))
            circles_st = self[i]
            # Add to list if non-empty
            nr_boundaries = len(boundaries_st)
            if nr_boundaries>0:
                boundaries_l.append(boundaries_st)
                circles_l.append(circles_st)
        return boundaries_l, circles_l
    
    def _orderBoundaries(self, boundaries, inner=False, prec=8):
        '''
        Args:
            boundaries: All intersections which lie on the innner or outer 
                boundary of the cluster of Circles.
            inner: Boolean indicating if we are ordering inner boundaries.
            prec: The number of decimals for which the comparisons are made on.
            
        Return:
            The sorted boundaries, i.e. the intersections Points lying on the
            boundary of the Circles as well as the corresponding Circles
            and angles.
        '''
        boundaries_l, circles_l = boundaries

        # Variable indicating first loop
        first = True
        # Create receptacle lists.
        ordered_boundaries_l = []
        ordered_circles_l = []
        ordered_angles_l = []
        remaining_boundaries_l = []
        remaining_circles_l = []
        
        '''
        # As a starting points for finding the ordered boundaries we first find 
        #  the index  of the Circles which have boundaries (i.e. intersections on 
        #  the boundary) with the lowest overall x-value (can be more than 1).
        xmin = min([_.x.min().round(prec) for _ in boundaries_l])
        xmin_index = [i for i,_ in enumerate(boundaries_l) if _.x.min().round(prec) == xmin]
        # From those Circles we look which one has a boundary with the highest 
        #  y-values (can be more than 1).
        ymax_xmin = [boundaries_l[_].y.max().round(prec) for _ in xmin_index]
        xmin_subindex = [i for i,_ in enumerate(ymax_xmin) if _ == max(ymax_xmin)]
        xmin_index = [xmin_index[_] for _ in xmin_subindex]
        # From those Circles we look which one has the overall lowest y-value
        xmin_xmin = [(circles_l[_].x-circles_l[_].r).round(prec) for _ in xmin_index]
        xmin_subindex = int(np.array([i for i,_ in enumerate(xmin_xmin ) if _ == min(xmin_xmin)]))
        xmin_index = xmin_index[xmin_subindex]
        previous_i = xmin_index
        # The boundary from which our discovery process of ordered boundaries 
        # starts, is the one with the lowest x.
        boundary = [_ for _ in boundaries_l[xmin_index] if _.x.round(prec) == xmin][0]
        '''
        xmin = min([(_.x-_.r).round(prec) for _ in circles_l])
        xmin_index = [i for i,_ in enumerate(circles_l) if (_.x-_.r).round(prec) == xmin][0]
        previous_i = xmin_index
        xmin_intersects = min(boundaries_l[xmin_index].y.round(prec))
        boundary = [_ for _ in boundaries_l[xmin_index] if _.y.round(prec) == xmin_intersects][0]
        # We define the according Circle (needed as return) and its centerpoint
        #  (needed for the method's calculations).
        c = circles_l[xmin_index]
        cp = c.xy
        
        # We loop indefinitely...
        while True: 
            # i indicates the index of the Circle within our cluster from which
            #  the next boundary will be taken from (to add to our ordered boundaries 
            #  list). We define i as those indices of the Circles which have boundaries
            #  that are the same as the previous boundary added to the ordered
            #  boundaries list.
            i = ([i for i,_ in enumerate(boundaries_l) 
                  if np.any(np.all(_.round(prec)==boundary.round(prec), axis=1))])
            # We allways favor an index different from the index from which 
            #  the last boundary was taken from, unless there is no other choice...
            i = [_ for _ in i if _ != previous_i][0] if len(i)>1 else i[0]
            # ...or we are looking for our first contribution; Then we look for our
            #  next (or rather first) addition to our ordered boundaries from the 
            #  Circle from which the starting point was taken from.
            if first:
                i = previous_i
            # The candidates as next addition are all the boundaries from the 
            #  Circle with index i (look above).
            bp = boundaries_l[i]
            # We define the according Circle, centerpoint and number of boundaries/candidates.
            c = circles_l[i]
            cp = c.xy
            n_bounds = len(bp)
            # Note that with defining "i" we have correcly found the Circle from
            #  which the next boundary should be selected from. We now identify the
            #  index of the individual boundary from those boundaries which match
            #  the previous addition.
            mask = (bp.round(prec)==boundary.round(prec)).all(axis=1)
            j = int(np.arange(n_bounds)[mask])
            # From the boundaries/candidates we now exclude the boundary which
            #  matches the previous addition, leaving us with one or multiple
            #  boundaries to choose from as next addition.
            ordered_remainders, angles = bp.drop(j).orderedPoints(cp, bp[j], True)
            if inner:
                angle = max(angles)
                # The next boundary is found by taking the last bp of the ordered bp's
                boundary = ordered_remainders[-1]
            else:
                angle = min(angles)
                # The next boundary is found by taking the first bp of the ordered bp's
                boundary = ordered_remainders[0]
            # We break the loop if the ordered boundaries loop is closed, i.e.
            #  if the first ever addition mathces the last one made.
            if not first and np.all(ordered_boundaries_l[0].round(prec)==boundary.round(prec)):
                break
            # Add it to the list as well as the corresponding Circle and angle.
            ordered_boundaries_l.append(boundary)
            ordered_circles_l.append(c)
            ordered_angles_l.append(angle)
            # Reset previous i.
            previous_i = i
            first = False
        # Collect boundaries which have note been used in the closed circuit, 
        #  i.e. boundaries of 'inner holes'
        try:
            # We try to identify the boundaries (and their corresponding indices) 
            #  which have not been used in the orderd boundaries list.
            # Those boundaries are boundaries of an (or a next) inner hole.
            indices, candidates = zip(*[(i,p) for (i,sublist) in enumerate(boundaries_l) for p in sublist 
                                        if (p.round(prec) != np.array(ordered_boundaries_l).round(prec)).all()])
            remaining_circles = Circle([circles_l[_] for _ in indices]).round(prec)
            # To preserve the same structure of Boundaries - Circle, we identify
            #  all boundaries for each specific Circle.
            for u_c in np.unique(remaining_circles, axis=0):
                mask = (remaining_circles== u_c).all(axis=1)
                p = Point([candidates])[mask]
                c = Circle([u_c])
                remaining_boundaries_l.append(p)
                remaining_circles_l.append(c)
        except ValueError:
            warnings.warn('No boundaries left to categorize.')
            
        return (ordered_boundaries_l,
                ordered_circles_l,
                ordered_angles_l,
                remaining_boundaries_l,
                remaining_circles_l)
        
    def calc_boundaries(self):
        '''
        Returns:
            - The outer boundaries of the Circle-cluster in a clockwise order as well
              as the corresponding circles and angles.
            - The inner boundaries of the Circle-cluster in a grouped and clockwise
              order as well as the corresponding circles.
        '''
        # Test if the Circles are clustered.
        if self.iscluster == False:
            raise(RuntimeError, "Boundaries should be computed for a specific cluster.")
        # Indicate boundaries have been computed.
        self.isbounded = True
        # Test if the cluster exists of only 2 or less Circles.
        if len(self)<3:
            return None
        # Compute all the available boundaries.
        boundaries_to_order =  self._all_boundaries()
        # Group and order all the boundaries on being outer boundaries.
        ordered_b, ordered_c, ordered_a, remaining_b, remaining_c = self._orderBoundaries(boundaries_to_order)
        self.outer_boundaries = ordered_b, ordered_c, ordered_a
        boundaries_to_order = remaining_b, remaining_c
        # Group and order all the remaining (inner) boundaries.
        inner_boundaries_l = []
        while len(boundaries_to_order[0])>0:
            ordered_b, ordered_c, ordered_a, remaining_b, remaining_c = self._orderBoundaries(boundaries_to_order, inner=True)
            inner_boundaries_l.append((ordered_b,ordered_c))
            boundaries_to_order = remaining_b, remaining_c
        self.isbounded = True
        self.inner_boundaries = inner_boundaries_l
   
    def intersectCord(self, circle):
        '''
        Args:
            circle: The other Circle with whom the cord is created.
        
        Returns:
            The length of the intersecting cord.
        '''
        d = self.distance(circle)
        r0 = self.r
        r1 = circle.r
        a = (1/d)*math.sqrt((-d+r1-r0)*(-d-r1+r0)*(-d+r1+r0)*(d+r1+r0))
        return a
        
    @staticmethod
    def _circularSegment(r, cord):
        '''
        Args:
            r: The radius of the first Circle.
            cord: The length of the cord for which we want the circular segment.
        
        Returns:
            The circular segment delimited by the cord.
        '''
        a = cord
        bR = r
        sr = (1/2)*math.sqrt(4*bR**2-a**2)
        h = bR-sr
        bA = bR**2 * math.acos((bR-h)/bR) - (bR-h)*math.sqrt(2*bR*h-h**2)
        return bA
    
    def intersectArea(self , circle, show_segments=False):
        '''
        Args:
            circle: One circle of which we want to know the intersecting area 
                off with self.
            show_segments: Boolean indicating if the individual contributions 
                to the total Area need to be returned.
        
        Returns:
            The intersecting area of two Circles.
        '''
        a = self.intersectCord(circle)
        A_self = self._circularSegment(self.r, a)
        A_circle_2 = circle.circularSegment(a)
        A_total = A_self+A_circle_2
        if show_segments:
            return [A_total,[A_self,A_circle_2]]
        else:
            return A_total
    
    def flatArea(self):
        '''
        The exact method for computing the area of a cluster of Circles.
        
        Returns:
            The Area of the cluster of Circles taking into account:
                - overlapping areas which are added only once.
                - inner holes of the cluster of Circles which are substracted.
        '''
        # Test if boundaries have been computed.
        if self.isbounded == False:
            raise(RuntimeError, "Boundaries should be computed in order to compute flatArea.")
        # Case for a cluster of 1 Circle.
        if len(self)==1:
            A = self.area()
            return A
        # Case for a cluster of 2 Circles.
        if len(self)==2:
            A = self[0].intersectArea(self[1])
            return A
        # Case for a cluster of 3 or more Circles.
        if len(self)>2:
            # 1. We compute the Polygon area.
            ordered_b, ordered_c, ordered_a = self.outer_boundaries
            ordered_cp = [_.xy for _ in ordered_c]
            ordered_all = np.hstack((ordered_cp, ordered_b)).reshape(-1,2)
            ordered_all = np.append(ordered_all, ordered_all[0].reshape(-1,2), axis=0)
            ordered_all = Point(ordered_all)
            polygon_A = ordered_all.polygonArea()
            # 2. We compute the Area shaved of the Circles when defining the polygon.
            shaved_A = sum((np.array(ordered_a)/360)*Circle(ordered_c).area())
            # 3. We compute the included inner hole area(s) that need to be removed.
            all_holes_l = []
            inner_boundaries = self.inner_boundaries.copy() 
            for ordered_b, ordered_c in inner_boundaries: # The can be more than 1 hole.
                ordered_all = ordered_b+ordered_b[0]
                ordered_all = Point(ordered_all)
                # 3.a. Inner polygon area
                inner_polygon_A = ordered_all.polygonArea()
                # 3.b. Included shaved area that needs to be removed from the inner polygon area.
                cords = [ordered_all[i].distance(ordered_all[i+1]) for i in range(len(ordered_all)-1)]
                inner_shaved_A_l = [self._circularSegment(ordered_c[i].r, cords[i]) for i in range(len(cords))]
                inner_shaved_A = sum(inner_shaved_A_l)
                # Total area inner hole
                inner_hole_A = inner_polygon_A - inner_shaved_A
                all_holes_l.append(inner_hole_A)
            all_holes_A = sum(all_holes_l)
            # Total area
            return polygon_A + shaved_A - all_holes_A 
    
    def simArea(self, n_samples=None):
        '''
        Returns:
            The approximated area using monte carlo simulations.
        '''
        xmin = min(self.x - self.r)
        xmax = max(self.x + self.r)
        ymin = min(self.y - self.r)
        ymax = max(self.y + self.r)
        random_area = (xmax-xmin)*(ymax-ymin)
        if n_samples==None:
            n_samples = int(random_area*200)
        random_p =  Point.random(xmin, xmax, ymin, ymax, n_samples)        
        result = np.array([any(self.encompass(_)) for _ in random_p]).astype(int)
        mean = result.mean()
        approx_area = random_area*mean
        return approx_area