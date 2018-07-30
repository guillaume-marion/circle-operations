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
    Note the difference in notation between:
        - 'circle' which at least represent a set of coordinates+radius in a list, array, ...
        - 'Circle' which represents an instance of the class.
    '''
    
      ###############################
     #### Dunder and properties ####
    ###############################
    
    def __new__(cls, inputarray):
        '''
        A Circle is created from a np.ndarray.
        The array can exist of a single set of coordinates+radius or multiple, e.g.:
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
        self._isclustered = False
        self._iscluster = False
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
            nr_circles: The number of xy-coordinates & radii to be produced.
        
        Returns: 
            Random xy-coordinates & radii.
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
            nr_circles: The number of xy-coordinates & radii to be produced.
        
        Returns: 
            A random instance of a Circle.
        '''
        xyr_values = cls._random(x_min, x_max, y_min, y_max, radius_min, radius_max, nr_circles)
        random_circle = cls(xyr_values)
        return random_circle
    
    @classmethod
    def _populate_lines(cls, m_point, nr_circles, jitter_sd, radius_min, radius_max):
        '''
        Args:
            m_point: Multiple Points.
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
        populated_lines = super(Circle, cls)._populate_lines(m_point, nr_circles, jitter_sd)
        random_radii = np.random.uniform(radius_min, radius_max, len(populated_lines))
        random_radii = random_radii.reshape(-1,1)
        populated_lines = np.append(populated_lines, random_radii, axis=1)
        return populated_lines
    
    @classmethod
    def populate_lines(cls, point, nr_circles, jitter_sd=1, radius_min=2, radius_max=2):
        '''
        Args:
            point: Multiple Points.
            nr_circles: The number of circles to be created for each line.
            jitter_sd: The standard deviation of the normal distribution
                from which the jitter is sampled.
            radius_min: The minimum value for the radii.
            radius_max: The maximum value for the radii.
        Returns:
            A number of randomized Circles between each set of Points in their 
            given order. This allows one to populate Circles along a path defined
            by a set of given Points.
        '''
        populated_lines = cls._populate_lines(point, nr_circles, jitter_sd, radius_min, radius_max)
        populated_lines_as_Circles = Circle(populated_lines)
        return populated_lines_as_Circles
    
    
      ######################
     #### Core methods ####
    ######################
    
    def area(self):
        '''
        Returns: The area(s) as a numpy.ndarray.
        '''
        return math.pi*self.r**2
    
    @staticmethod
    def _intersect(s_circle_r, s_circle_xy, m_circle_r, m_circle_xy, distance):
        '''
        Args:
            s_circle_r: A single radius as a numpy.ndarray.
            s_circle_xy: A single set of xy-coordinates as a numpy.ndarray.
            m_circle_r: A single or multiple sets of radii as a numpy.ndarray.
            m_circle_xy: A single or multiple sets of xy-coordinates as a
                numpy.ndarray.
        
        Returns: The xy-coordinates of the intersection points between s_circle 
            and m_circle as a numpy.ndarray.
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
            # If only non-intersecting circles are provided, return None.
            #if mask.sum()==len(r1):
            #    return None
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
    
    def intersect(self, circle):
        '''
        Args:
            circle_or_list: One or multiple Circles.
        
        Returns: The xy-coordinates of the intersection points between the
            origin Circle and the (multiple) circle(s) as a tuple of 2 Points.
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
        Returns: The xy-coordinates of the intersection points between each of
            the origin Circles and the other origin Circles as a list of tulpes
            of 2 Points.
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
            # Store results
            self.intersections = intersects_l
    
    def encompass(self, s_point_or_circle, prec=8):
        '''
        Args:
            s_point_or_circle: A single Point or Circle to test.
        
        Returns:
            Boolean indicating if the single Point or Circle is encompassed.
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
            is_in_circle = ((x-h)**2+(y-k)**2).round(prec)<(r**2).round(prec)
            return is_in_circle
    
    def calc_clusters(self):
        '''
        Return: 
            A tuple of which each element is a set of indices of an 
            intersecting group of Circles.
        '''
        # If not calculated yet, get all the intersections (many-to-many).
        if len(self.intersections) == 0:
            raise(RuntimeError, "To compute clusters, compute intersections first.")
        # Define constants
        nr_circles = len(self.intersections)
        index = np.arange(nr_circles)
        # Define receptacles.
        intersections_indices = []
        nr_intersections = []
        # Retrieve for each Circle the indices of the other Circles which intersect.
        #  Also add the index of the Circle under consideration.
        for i,_ in enumerate(self.intersections):
            mask = np.isnan(_[0]).all(axis=1)==False
            indices = set(np.delete(index, i)[mask])
            nr_intersections.append(len(indices))
            indices.add(i)
            intersections_indices.append(indices)
        # Provide a starting point for finding the Circle-clusters:
        #  Here we just take all the indices of the Circles intersecting with 
        #  the first Circle (i.e. index=0).
        first_cluster_contribution = intersections_indices[0]
        cluster_list =  [first_cluster_contribution]
        # We identify the clusters by adding the indices-groups to a cluster if 
        #  at least one index within the group matches the cluster. In other words,
        #  if A intersects with B, C and K, then it forms a basis for a cluster 
        #  A-B-C-K (i.e. the aformentioned kickstart). Now if X intersects with 
        #  Y, Z and K then we add it to the cluster , which becomes A-B-C-X-Y-Z-K.
        for i, cluster in enumerate(cluster_list): 
            final_cluster = cluster
            # List of indices of Circles that intersect with one another.
            #  E.g. remaining_indices[0] are all the indices of the Circles which
            #  intersect with Circle 0 (as well as the index 0).
            remaining_indices = intersections_indices.copy()
            # Continue until no new cluster is formed (controlled with "added").
            while True:
                added = False
                # For the construction of the cluster under consideration iterate
                #  over all the lists of indices that intersect with each other.
                for indices in remaining_indices:
                    # Add the indices to the cluster if any of the indices can be
                    #  found in the current cluster definition.
                    if len(final_cluster & indices)>0:
                        final_cluster = final_cluster | indices
                        remaining_indices.remove(indices)
                        added = True
                if not added:
                    cluster_list[i] = final_cluster
                    break
            # We create a new cluster (or rather a kickstart for a cluster) with
            #  an indices-group of which none of the indices can be found in any
            #  of the previously constructed clusters.
            all_indices = set([item for sublist in cluster_list for item in sublist])
            new_clusters_candidates = ([sett for sett in remaining_indices 
                                        if len(sett & all_indices)==0 and len(sett)>0])          
            if len(new_clusters_candidates)>0:
                cluster_list.append(new_clusters_candidates[0])
            # Stop if no new cluster is created
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
        # Store results
        self._clusters_indices = cluster_groups
        self.nr_clusters = len(cluster_groups)
        self._isclustered = True
    
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
                indices: All the indices of the Circles within the cluster.
                self_index: The index of the Circle under consideration.
            
            Returns:
                A list of indices that can be used to slice only those intersections 
                between the Circle (with index = self_index) and the other Circles
                within the cluster.
                
            Note: This function is needed since the .intersect() method results in n-1
                intersections in the event of n Circles, therefore shifting the 
                index of the original Circles. It might be interseting to add an 
                intersection with self that equals to None, keeping the index unaltered.
            '''
            corrected_indices = []
            lower_than = [_ for _ in indices if _<self_index]
            corrected_indices.extend(lower_than)
            higher_than = [_-1 for _ in indices if _>self_index]
            corrected_indices.extend(higher_than)
            return corrected_indices
        
        # Assert clusters have been computed
        if self._isclustered == False:
            raise(RuntimeError, "To extract a cluster, compute clusters first.")
        # Get the specific set of indices as a list and sort
        indices_list = list(self._clusters_indices[cluster_index])
        indices_list.sort()
        # Create a new Circle of the relevant portion of self 
        Clustered_circle = Circle(self[indices_list])
        # Get the intersections Points for the Circles in the cluster
        #  Note that we have to rework the Points to exclude all intersections
        #  between the Cluster's Circles and Circles not included in the cluster.
        #  We *could* just recompute the .intersections() method on the new instance. 
        all_intersections = [self.intersections[_] for _ in indices_list]
        lean_intersections = []
        for i,index in enumerate(indices_list):
            i1,i2 = all_intersections[i]
            int_indices = _intersect_indices(indices_list, index)
            lean_intersections.append((i1[int_indices],i2[int_indices]))
        # Redefine variables
        Clustered_circle.intersections = lean_intersections
        Clustered_circle._iscluster = True
        Clustered_circle._isclustered = True # Needed because new instance...
        Clustered_circle.nr_clusters = 1
        return Clustered_circle
        
    
    def _boundaries(self):
        '''
        Returns: The xy-coordinates of the intersection points between each of
            the origin Circles and the other origin Circles as a Point as well as
            the corresponding centerpoint. The result is returned only if the 
            aformentioned intersections are not encompassed by a Circle, i.e. 
            the intersections lies on the boundary of the Circles.
        '''
        # Assert if the Circles are clustered
        if self._iscluster == False:
            raise(RuntimeError, "Boundaries should be computed for a specific cluster")
        boundaries_l = []
        centerpoints_l = []
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
            centerpoints_st = self[i].xy
            # Add to list if non-empty
            nr_boundaries = len(boundaries_st)
            if nr_boundaries>0:
                boundaries_l.append(boundaries_st)
                centerpoints_l.append(centerpoints_st)
        return boundaries_l, centerpoints_l
    
    def orderedBoundaries(self, prec=8):
        '''
        Args:
            prec: The number of decimals for which the comparisons are made on.
            
        Return:
            The sorted boundaries, i.e. the intersections Points lying on the
            boundary of the Circles as well as the corresponding center Points.
        '''
        boundaries_l, centerpoints_l = self._boundaries()

        # Params
        first = True
        ordered_boundaries_l = []
        ordered_centerpoints_l = []
        ordered_angles_l = []
        remaining_boundaries_l = []
        remaining_centerpoints_l = []
        
        # Starting point = of the Circles with the leftmost intersections, we select
        #  that Circle which also has the upmost intersection. From that Circle we
        #  take the leftmost intersection as starting point.
        xmin = min([_.x.min().round(prec) for _ in boundaries_l])
        xmin_index = [i for i,_ in enumerate(boundaries_l) if _.x.min().round(prec) == xmin]
        ymax_xmin = [boundaries_l[_].y.max().round(prec) for _ in xmin_index]
        xmin_subindex = [i for i,_ in enumerate(ymax_xmin) if _ == max(ymax_xmin)][0]
        xmin_index = xmin_index[xmin_subindex]
        previous_i = xmin_index
        boundary = [_ for _ in boundaries_l[xmin_index] if _.x.round(prec) == xmin][0]
        cp = centerpoints_l[xmin_index]
        
        while True: 
            # Favor new centerpoint
            i = ([i for i,_ in enumerate(boundaries_l) 
                  if np.any(_.round(prec)==boundary.round(prec))])
            i = [_ for _ in i if _ != previous_i][0] if len(i)>1 else i[0]
            if first:
                i = previous_i
            bp = boundaries_l[i]
            cp = centerpoints_l[i]
            n_bounds = len(bp)
            # Find the index of the boundary in the bp's
            mask = (bp.round(prec)==boundary.round(prec)).all(axis=1)
            j = int(np.arange(n_bounds)[mask])
            # Order the bp's excluding the previous boundary
            ordered_remainders, angles = bp.drop(j).orderedPoints(cp, bp[j], True)
            angle = min(angles)
            # The next boundary is found by taking the first bp of the ordered bp's
            boundary = ordered_remainders[0]
            # Break if boundary is closed
            if not first and np.all(ordered_boundaries_l[0].round(prec)==boundary.round(prec)):
                break
            # Add it to the list as well as the corresponding cp
            ordered_boundaries_l.append(boundary)
            ordered_centerpoints_l.append(cp)
            ordered_angles_l.append(angle)
            # Reset previous i
            previous_i = i
            first = False
        # Collect boundaries which have note been used in the closed circuit, 
        #  i.e. boundaries of 'inner holes'
        try:
            indices, candidates = zip(*[(i,p) for (i,sublist) in enumerate(boundaries_l) for p in sublist 
                                        if (p.round(prec) != np.array(ordered_boundaries_l).round(prec)).all()])
            remaining_centerpoints = Point([centerpoints_l[_] for _ in indices]).round(prec) 
            for u_cp in np.unique(remaining_centerpoints, axis=0):
                mask = (remaining_centerpoints == u_cp).all(axis=1)
                p = Point([candidates])[mask]
                cp = Point([u_cp])
                remaining_boundaries_l.append(p)
                remaining_centerpoints_l.append(cp)
        except ValueError:
            warnings.warn('no inner boundaries')

        return ordered_boundaries_l, ordered_centerpoints_l, ordered_angles_l, remaining_boundaries_l, remaining_centerpoints_l 
        
    
      ##########################
     #### Obsolete methods ####
    ##########################

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
    