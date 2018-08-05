
    @classmethod
    def _populate_lines(cls, m_point, nr_points, jitter_sd):
        '''
        Args:
            m_point: At least two Points.
            nr_points: The number of points to be created for each line.
            jitter_sd: The standard deviation of the normal distribution
                from which the the jitter is sampled.
                
        Returns:
            A number of randomized points between each set of Points in their 
            given order. This allows one to populate points along a path defined
            by a set of given Points.
        '''
        populated_lines = np.array([]).reshape(0,2)
        nr_segments = len(m_point)-1
        
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
                Randomized points on the line between the 2 Points. 
                Also some jitter (based on a normal distribution) is
                added as vertical distance from the line.
            '''
            # Get coefficient and intercept of the line between the two Points.
            a,b = _lineParameters(s_point, s_point_2)
            # Get the mininmum and maximum values of the two Points.
            x_min, x_max = _xRange(s_point, s_point_2)
            # Produce random x-values within the calculated range.
            x = np.random.uniform(x_min, x_max, nr_points)
            # Produce normal-random jitter.
            jitter_values = np.random.normal(0, jitter_sd, nr_points)
            # Compute corresponding y-values based on the line-equation on which
            #  the normal-random vertical jitter is added.
            y = (a*x) + b + jitter_values
            # Return the coordinates as a np.ndarray.
            return np.array(list(zip(x,y)))
    
        # For each set of two Points (i.e. a segment) produce points on the line.
        for i in range(nr_segments):
            populated_line = _populate_line(m_point[i], m_point[i+1], nr_points, jitter_sd)
            populated_lines = np.append(populated_line, populated_lines, axis=0)
        
        return populated_lines
    
    @classmethod
    def populate_lines(cls, point, nr_points, jitter_sd=1):
        '''
        Args:
            point: At least two Points.
            nr_points: The number of Points to be created.
            jitter_sd: The standard deviation of the normal distribution
                from which the the jitter is sampled.
                
        Returns:
            A number of randomized Points between each set of Points in their 
            given order. This allows one to populate Points along a path defined
            by a set of given Points.
        '''
        # Produce points with the given args.
        populated_lines = cls._populate_lines(point, nr_points, jitter_sd)
        # Initialize the points as Points.
        populated_lines_as_Points = Point(populated_lines)
        
        return populated_lines_as_Points



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
    