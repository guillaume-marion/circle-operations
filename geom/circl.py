# Local imports
from geom.pnt import Point

# Library imports
import math
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
        removed = set()
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
                    break
            # Within our final cluster we remove any Circle which is encompassed
            #  by any of the other Circles within the cluster.
            to_remove = [cl for cl in final_cluster if self[list(final_cluster)].encompass(self[int(cl)])]
            removed = removed | set(to_remove)
            cluster_list[i] = final_cluster - set(to_remove)
            # For the loop to continue we need a proposal for a next cluster.
            # We create a said proposal with an indices-group (i.e. a Circle's index
            #  as well as its intersection's indices) of which none of the indices 
            #  can be found in any of the previously constructed clusters.
            all_indices = set([item for sublist in cluster_list for item in sublist])
            all_indices_plus = all_indices | removed
            new_clusters_candidates = ([sett for sett in remaining_indices 
                                        if (len(sett & all_indices_plus)==0 and len(sett)>0)])          
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
        final_clusters = []
        # We do not include encompassed Circles in the clusters definition, as
        #  these Circles do not add any information to area calculations.
        for isolat in cluster_isolats:
            encompassment = self.encompass(self[list(isolat)])
            if encompassment == False:
                final_clusters.append(isolat)
        # We do not include encompassed clusters in the clusters definition, as
        #  these clusters do not add any information to area calculations.
        for cluster_group in cluster_groups:
            encompassment = all([self.encompass(_) for _ in self[list(cluster_group)]])
            if encompassment == False:
                final_clusters.append(cluster_group)
        # Store results.
        self._clusters_indices = final_clusters
        self.nr_clusters = len(final_clusters)
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
    
    def _orderBoundaries(self, boundaries, inner=False, start_index=None, prec=8):
        '''
        Args:
            boundaries: All intersections which lie on the innner or outer 
                boundary of the cluster of Circles.
            inner: Boolean indicating if we are ordering inner boundaries.
            prec: The number of decimals for which the comparisons are made on.
            index: The index of the starting point.
            
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
        remaining_boundaries_l = []
        remaining_circles_l = []
        # If a starting index is provided (for the Circle and one of its 
        #  boundaries) then use that starting index.
        if start_index:
            start_cindex, start_bindex = start_index
            boundary = boundaries_l[start_cindex][start_bindex]
            previous_i = start_cindex
        # If no starting index is provided, we make a best guess for a start.
        else:
            # As a starting points for finding the ordered boundaries we first find 
            #  the index of the Circle with the boundary having the lowest x-value. 
            xmin = min([(_.x).round(prec).min() for _ in boundaries_l])
            start_cindex_l = [i for i,_ in enumerate(boundaries_l) if (_.x).round(prec).min() == xmin]
            # We pick the Circle with the highest centerpoint (usefull for 
            #  inner boundaries)
            random_subindex = np.random.randint(len(start_cindex_l))
            start_cindex = start_cindex_l[random_subindex ]
            previous_i = start_cindex
            # The boundary from which our discovery process of ordered boundaries 
            #  then starts, is the one with the lowest y.
            ymin = min(boundaries_l[start_cindex].y.round(prec))
            boundary = [_ for _ in boundaries_l[start_cindex] if _.y.round(prec) == ymin][0]

        # We define the according Circle (needed as return) and its centerpoint
        #  (needed for the method's calculations).
        c = circles_l[start_cindex]
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
                # The next boundary is found by taking the last bp of the ordered bp's
                boundary = ordered_remainders[-1]
                # For the first addidition we take a look at the angle.
                if first:
                    max_angle = max(angles)
                    # If the angle is smaller than 180° this indicates a wrong start
                    #  and we return None so that the method can be repeated with
                    #  another index-seed.
                    if max_angle < 270:
                        return None
            else:
                # The next boundary is found by taking the first bp of the ordered bp's
                boundary = ordered_remainders[0]
            # We break the loop if the ordered boundaries loop is closed, i.e.
            #  if the first ever addition mathces the last one made.
            if not first and np.all(ordered_boundaries_l[0].round(prec)==boundary.round(prec)):
                break
            # Add it to the list as well as the corresponding Circle and angle.
            ordered_boundaries_l.append(boundary)
            ordered_circles_l.append(c)
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
            pass
            
        return (ordered_boundaries_l,
                ordered_circles_l,
                remaining_boundaries_l,
                remaining_circles_l)
        
    def calc_boundaries(self, silent=True):
        '''
        Returns:
            - The outer boundaries of the Circle-cluster in a clockwise order as well
              as the corresponding circles.
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
        
        ########################
        ### OUTER BOUNDARIES ###
        ########################
        # Loop indefinitely unless...
        indices_combos = [(i,j) for (i,sublist) in enumerate(boundaries_to_order[0]) 
                                for (j,item) in enumerate(sublist)]
        counter = -1
        while True:
            selected_index = indices_combos[counter] if counter>-1 else None
            # Group and order all the boundaries on being outer boundaries.
            ordered_boundaries = self._orderBoundaries(boundaries_to_order, start_index=selected_index)
            ordered_b, ordered_c, remaining_b, remaining_c = ordered_boundaries 
            # Check if the ordered boundaries result in a polygon that encompasses 
            #  all boundaries.
            polygon = round(Point([ordered_b[-1]]+ordered_b),8)
            polygon_encompall = all([polygon.polyEncompass(round(_,8)) for _ in boundaries_to_order[0]])
            #... the created polygon encompasses all boundaries.
            if polygon_encompall:
                break
            # If not, we specify another start from the possible starts.
            counter += 1
        # Print information about the outer boundary discovery process.
        if not silent:
            print("Found a correct outer boundary after {} attempt(s).".format(counter+2))
        # Store the results
        self.outer_boundaries = ordered_b, ordered_c
        boundaries_to_order = remaining_b, remaining_c
        
        ########################
        ### INNER BOUNDARIES ###
        ########################
        inner_boundaries_l = []
        # Loop as long as there are (inner) boundaries to order.
        while len(boundaries_to_order[0])>0:
            # Keep looking for an inner boundary until...
            counter = -1
            indices_combos = [(i,j) for (i,sublist) in enumerate(boundaries_to_order[0]) 
                                    for (j,item) in enumerate(sublist)]
            while True:
                selected_index = indices_combos[counter] if counter>-1 else None
                # Apply the method.
                ordered_boundaries = self._orderBoundaries(boundaries_to_order, inner=True, start_index=selected_index)
                #...there is actually a result.
                if ordered_boundaries != None:
                    break
                # If not, we specify another start from the possible starts.
                counter += 1
            # Print information about the innner boundary discovery process.
            if not silent:
                print("(Re)computed inner boundary {} time(s).".format(counter+2))
            # Store the inner boundaries results.
            ordered_b, ordered_c, remaining_b, remaining_c = ordered_boundaries
            inner_boundaries_l.append((ordered_b,ordered_c))
            boundaries_to_order = remaining_b, remaining_c
        self.isbounded = True
        self.inner_boundaries = inner_boundaries_l
   
    def intersectChord(self, circle):
        '''
        Args:
            circle: The other Circle with whom the chord is created.
        
        Returns:
            The length of the intersecting chord.
        '''
        d = self.distance(circle)
        r0 = self.r
        r1 = circle.r
        a = (1/d)*math.sqrt((-d+r1-r0)*(-d-r1+r0)*(-d+r1+r0)*(d+r1+r0))
        return a
        
    @staticmethod
    def _circularSegment(r, chord):
        '''
        Args:
            r: The radius of the first Circle.
            chord: The length of the chord for which we want the circular segment.
        
        Returns:
            The circular segment delimited by the chord.
        '''
        a = chord
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
        a = self.intersectChord(circle)
        A_self = self._circularSegment(self.r, a)
        A_circle_2 = circle._circularSegment(circle.r, a)
        A_total = A_self+A_circle_2
        if show_segments:
            return [A_total,[A_self,A_circle_2]]
        else:
            return A_total
    
    def flatArea(self, return_edge_cases=False):
        '''
        The exact method for computing the area of one cluster of Circles.
        
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
            return float(A)
        
        # Case for a cluster of 2 Circles.
        if len(self)==2:
            circ_A = sum(self.area())
            intersect_A = self[0].intersectArea(self[1])
            return float(circ_A - intersect_A)
        
        # Case for a cluster of 3 or more Circles.
        if len(self)>2:
            ##############################################
            ### 1. We compute the outer boundary area. ###
            ##############################################
            ### 1.a. We compute the polygon area.
            # We get the boundaries & corresponding circles.
            ordered_b, ordered_c = self.outer_boundaries
            # We close the loop.
            ordered_all = Point([ordered_b[-1]]+ordered_b)
            # Compute the area of the polygon.
            polygon_A = ordered_all.polygonArea()
            
            ### 1.b. We compute the Area shaved of the Circles when defining the polygon.
            # Get the ordered centerpoints.
            ordered_cp = [_.xy for _ in ordered_c]
            # Test each centerpoint for encompassment by the polygon.
            cp_in_polygon = np.array([ordered_all.polyEncompass(_) for _ in ordered_cp])
            cp_notin_polygon = cp_in_polygon==False
            # Create the segments from the closed loop boundaries.
            segments = [ordered_all[[i,i+1]] for i in range(len(ordered_all)-1)]
            # We identify on which side the centerpoints lies of the segment (outwards or in).
            centroids = Point([_.centroid() for _ in segments])
            distances = [float(ordered_cp[i].distance(_)) for i,_ in enumerate(centroids)] # This should actually be distances to the line-segment and not the centroid!
            min_distances = [_.distance(centroids).min() for _ in ordered_cp]
            cp_outwards = np.array(distances)<=np.array(min_distances)
            # Combine both rules and keep track of those edge case cp's.
            mask = (cp_notin_polygon) & (cp_outwards)
            outer_edge_cp = Point([ordered_cp])[mask]
            # Compute the circular segments
            chords = [segment[0].distance(segment[1]) for segment in segments]
            circ_segments_A = [self._circularSegment(ordered_c[i].r, chords[i])[0] for i in range(len(chords))]
            circ_segments_A = np.array(circ_segments_A)
            # Compute the Circles' area.
            circ_A = Circle(ordered_c).area()
            # Apply the mask: if the centerpoint is not in the polygon and faces 
            #  outwards of the segment, then the area that needs to be added is 
            #  equal to the Circle's area - the circular segment. Otherwise the 
            #  area to be added is equal to the circular segment only.
            shaved_A_p1 = sum(circ_A[mask] - circ_segments_A[mask])
            shaved_A_p2 = sum(circ_segments_A[mask==False])
            shaved_A = shaved_A_p1 + shaved_A_p2
            
            ### 1.c. Total area outer boundary.
            outer_A = polygon_A + shaved_A
            
            ###################################################################
            ### 2. We compute the inner hole area(s) that need to be removed. #
            ###################################################################
            # For each set of inner boundaries.
            all_holes_l = []
            inner_edge_cp_l = []
            inner_boundaries = self.inner_boundaries.copy() 
            for ordered_b, ordered_c in inner_boundaries: # The can be more than 1 hole.
                ### 2.a. We compute the polygon area.
                # We close the loop.
                ordered_all = Point([ordered_b[-1]]+ordered_b)
                # Compute the area of the polygon.
                inner_polygon_A = ordered_all.polygonArea()
                
                ### 2.b. Included shaved area that needs to be removed from the inner polygon area.
                # Get the ordered centerpoints.
                ordered_cp = [_.xy for _ in ordered_c]
                # Test each centerpoint for encompassment by the polygon and keep track.
                cp_in_polygon = np.array([ordered_all.polyEncompass(_) for _ in ordered_cp])
                mask = cp_in_polygon
                inner_edge_cp_l.append(Point([ordered_cp])[mask])
                # Create the segments from the closed loop boundaries.
                segments = [ordered_all[[i,i+1]] for i in range(len(ordered_all)-1)]
                # Compute the circular segments
                chords = [segment[0].distance(segment[1]) for segment in segments]
                circ_segments_A = [self._circularSegment(ordered_c[i].r, chords[i])[0] for i in range(len(chords))]
                circ_segments_A = np.array(circ_segments_A)
                # Compute the Circles' area.
                circ_A = Circle(ordered_c).area()
                # Apply the mask: if the centerpoint is in the polygon then the
                #  area that needs to be substracted from the inner hole is equal 
                #  to the Circle's area - the circular segment. Otherwise the 
                #  area to be substracted is equal to the circular segment only.
                inner_shaved_A_p1 = sum(circ_A[mask] - circ_segments_A[mask])
                inner_shaved_A_p2 = sum(circ_segments_A[mask==False])
                inner_shaved_A = inner_shaved_A_p1 + inner_shaved_A_p2
                
                ### 2.c. Total area inner hole
                inner_hole_A = inner_polygon_A - inner_shaved_A
                all_holes_l.append(inner_hole_A)
            inner_edge_cp_l = [item for sublist in inner_edge_cp_l for item in sublist]
            inner_edge_cp = Point(inner_edge_cp_l) if len(inner_edge_cp_l)>0 else None
                
            all_holes_A = sum(all_holes_l)
            
            ##############
            # Total area #
            ##############
            if return_edge_cases:
                return float(outer_A - all_holes_A), outer_edge_cp, inner_edge_cp 
            else:
                return float(outer_A - all_holes_A)
    
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
        return float(approx_area)