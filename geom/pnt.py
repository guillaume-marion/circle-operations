# Library imports
import numpy as np

from geom.utils import atan2_v, degrees_v


# Defining the Point class
class Point(np.ndarray):
    """
    The Point class subclasses np.ndarray, for leveraging vectorized operations.
    Accordingly note that a Point can thus represent one or multiple points.
    """

    #############################
    #### Dunder & properties ####
    #############################

    def __new__(cls, inputarray):
        """
        A Point is created from a np.ndarray.
        The array can exist of a single or multiple points, e.g.:
        >>> p = Point([5,7])
        >>> mp = Point([[5,7],[13,4]])
        """
        obj = np.asarray(inputarray).view(cls)
        try:
            # Controlling for correct shape
            obj - np.asarray([1, 1])
            # Reshaping to columns
            obj = obj.reshape(-1, 2)
            return obj
        except:
            raise ValueError(
                "The input should have the shape of a (2,) or (*,2) array."
            )

    def __init__(self, inputarray):
        """
        We use the return from __new__ as self.
        """
        pass

    @property
    def x(self):
        return np.asarray(self[:, :1])

    @x.setter
    def x(self, value):
        self[:, :1] = value

    @property
    def y(self):
        return np.asarray(self[:, 1:2])

    @y.setter
    def y(self, value):
        self[:, 1:2] = value

    @property
    def xy(self):
        return self[:, :]

    @xy.setter
    def xy(self, value):
        self[:, :] = value

    @property
    def count(self):
        return self.shape[0]

    def __getitem__(self, val):
        """
        If val is an int, then consider it as a row-selection which should
            result in a Point. In every other case we use numpy's  __getitem__ method.
        Usecase: Selecting a single row from a multiple Point instance
            should also result in a Point with the correct shape (i.e. double
            enclosing brackets) to allow further use of the class' methods.
        """
        if type(val) == int:
            return self.__class__(np.asarray(self)[val])
        else:
            return super(Point, self).__getitem__(val)

    def __repr__(self):
        """
        Defines custom __repr__ method.
        """
        return str(self.__class__.__name__) + "([\n {:>10}\n])".format(
            self.__str__()[1:-1]
        )

    def __round__(self, decimals=0):
        """
        Defines custome __round__ method, which we can use to return a class
            compared to an array when using .round().
        """
        rounded_array = self.round(decimals)
        rounded_class = self.__class__(rounded_array)
        return rounded_class

    #############################
    #### Creation & Deletion ####
    #############################

    @staticmethod
    def _random(size, x_min, x_max, y_min, y_max):
        """
        Args:
            size: The number of points to be produced.
            x_min: Minium value for x-coordinates.
            x_max: Maximum value for x-coordinates.
            y_min: Minimum value for y-coordinates.
            y_max: Maximum value for y-coordinates.

        Returns:
            A np.ndarray of random points.
        """
        x = np.random.uniform(x_min, x_max, size)
        y = np.random.uniform(y_min, y_max, size)
        xy = np.dstack([x, y])
        return xy

    @classmethod
    def random(cls, size, x_min=0, x_max=10, y_min=0, y_max=10):
        """
        Args:
            size: The number of Points to be produced.
            x_min: Minium value for x-coordinates.
            x_max: Maximum value for x-coordinates.
            y_min: Minimum value for y-coordinates.
            y_max: Maximum value for y-coordinates.

        Returns:
            A random instance of (a) Point(s).
        """
        xy_values = cls._random(
            size=size, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
        random_point = cls(xy_values)
        return random_point

    def drop(self, row):
        """
        Args:
            row: The index of the row to be dropped.

        Returns:
            The Point without the specified row.
        """
        lower_end = self[:row]
        upper_end = self[row + 1 :]
        appended = np.append(lower_end, upper_end, axis=0)
        appended_point = self.__class__(appended)
        return appended_point

    def dropna(self):
        """
        Returns:
            The Point without nan values.
        """
        mask = np.array((np.isnan(self) == False).any(axis=1))
        self_without_nan = self[mask, :]
        return self_without_nan

    ##############
    #### Core ####
    ##############

    @staticmethod
    def _distance(px, py, mpx, mpy):
        """
        Args:
            px: One x-coordinate as a scalar or array.
            py: One y-coordinate as a scalar or array.
            mpx: One or multiple x-coordinates as an array.
            mpy: One or multiple y-coordinates as an array.

        Returns: The euclidean distance(s) between the point defined by [px, py] and the point(s) defined by [mpx, mpy].
        """
        return np.sqrt((mpx - px) ** 2 + (mpy - py) ** 2)

    def distance(self, other_points):
        """
        Args:
            other_point: another Point instance representing one (if self represents multiple points) or multiple points.

        Returns: The euclidean distance(s) between self and the other point(s).
        """
        assert (
            self.count == 1 or other_points.count == 1
        ), "When computing distances, only one of self or other_points can represent multiple points."
        if self.count == 1:
            p = self
            mp = other_points
        else:
            p = other_points
            mp = self
        return self._distance(
            px=p.x,
            py=p.y,
            mpx=mp.x,
            mpy=mp.y,
        )

    @staticmethod
    def _angle_to_align(px, py, mpx, mpy):
        """
        Args:
            px: One x-coordinate as a scalar or array.
            py: One y-coordinate as a scalar or array.
            mpx: One or multiple x-coordinates as an array.
            mpy: One or multiple y-coordinates as an array.

        Returns: The angle(s) at which (a) fully horizontal line(s) needs to rotate
            clockwise in order to match the line(s) between the point defined by [px, py] and the point(s) defined by [mpx, mpy].
        """
        dx = mpx - px
        dy = mpy - py
        return degrees_v(atan2_v(dy, dx))

    def angle_to_align(self, other_points):
        """
        Args:
            other_point: another Point instance representing one (if self represents multiple points) or multiple points.

        Returns: The angle(s), at which the horizontal line(s)
            needs to rotate clockwise in order to match the line(s) between self and
            the other point(s).
        """
        assert (
            self.count == 1 or other_points.count == 1
        ), "When computing angles, only one of self or other_points can represent multiple points.."
        if self.count == 1:
            p = self
            mp = other_points
        else:
            p = other_points
            mp = self
        return self._angle_to_align(
            px=p.x,
            py=p.y,
            mpx=mp.x,
            mpy=mp.y,
        )

    def angle_between(self, point, other_points):
        """
        Args:
            point: A single point from which the clockwise angle to the other point(s) is calculated.
            other_points: One or multiple points.

        Returns: The angles between the point and the other
            point(s), with self as vertex (vertices).
        """
        assert (
            point.count == 1
        ), "When computing angles, only one of self or other_points can represent multiple points."
        assert (
            self.count == 1 or other_points.count == 1
        ), "When computing angles, only one of self or other_points can represent multiple points."
        angleOrigin = 180 - self.angle_to_align(point)
        anglePoints = 180 - self.angle_to_align(other_points)
        anglePoints[anglePoints < angleOrigin] = (
            anglePoints[anglePoints < angleOrigin] + 360
        )
        return anglePoints - angleOrigin

    @staticmethod
    def _centroid(mpx, mpy):
        """
        Args:
            mpx: One or multiple x-coordinates as an array.
            mpy: One or multiple y-coordinates as an array..

        Returns: The centroid.
        """
        centroid = [mpx.sum() / mpx.size, mpy.sum() / mpx.size]
        return centroid

    def centroid(self):
        """
        Returns: The centroid of self as a Point.
        """
        centroid = self._centroid(mpx=self.x, mpy=self.y)
        return Point(centroid)

    def order_clockwise(self, start=None, center=None, return_angles=False):
        """
        Args:
            start: The starting point for computing the clockwise order. Defaults to the horizontal line as starting point.
            center: The center for the clock-pointer. Defaults to the centroid of self.
            return_angles: Whether or not to return the angles used to order the points.

        Returns:
            The Points in a clockwise-ordered fashion.
        """
        if center is None:
            centr = self.centroid()
        else:
            centr = center

        if start is None:
            angles = centr.angle_to_align(other_points=self) * -1
        else:
            angles = centr.angle_between(point=start, other_points=self)

        ordered_self = self[angles.argsort(axis=0)][:, 0]
        if return_angles:
            angles.sort()
            return ordered_self, angles
        else:
            return ordered_self


class Polygon(Point):
    """
    A Polygon is a collection of ordered Points.
    """

    #############################
    #### Dunder & properties ####
    #############################

    def __new__(cls, inputarray):
        """
        A Polygon is created from a np.ndarray.
        The array exists of multiple points, e.g.:
        >>> Polygon([[5,7],[13,4]])
        """
        obj = np.asarray(inputarray).view(cls)
        try:
            # Controlling for correct shape
            obj - np.asarray([1, 1])
            # Reshaping to columns
            obj = obj.reshape(-1, 2)
            # Controlling for minimum size
            assert obj.shape[0] > 2
            obj = obj.order_clockwise()
            return obj
        except:
            raise ValueError("The input should be an array of shape (>2, 2).")

    def __init__(self, inputarray):
        """
        We use the return from __new__ as self.
        """
        pass

    def __getitem__(self, val):
        """
        If val is an int or results in fewer than 3 points, return a Point instance.
        """
        # Return a Point instance if the selection has fewer than 3 rows
        if isinstance(val, int) or (
            isinstance(val, slice)
            and (
                (val.stop if val.stop is not None else self.shape[0])
                - (val.start if val.start is not None else 0)
            )
            < 3
        ):
            return Point(np.asarray(self)[val])
        else:
            return super(Point, self).__getitem__(val)

    @staticmethod
    def _area(mpx, mpy):
        """
        Args:
            mpx: Multiple x-coordinates in clockwise order.
            mpy: Multiple y-coordinates in clockwise order.

        Returns: The area of the polygon bounded by the points.
        """
        return 0.5 * np.abs(
            np.dot(mpx.T[0], np.roll(mpy.T[0], 1))
            - np.dot(mpy.T[0], np.roll(mpx.T[0], 1))
        )

    def area(self):
        """
        Returns:
            The area of the polygon.
        """
        area = self._area(self.x, self.y).item()
        return area

    def contains(self, points, return_points=True, include_vertices=True):
        """
        Args:
            points (Point): A Point instance with one or multiple points to check for containment.

        Returns:
            - Point instance containing only those points within the polygon.
        """
        # Ensure points is a Point instance
        if not isinstance(points, Point):
            raise TypeError("Input must be a Point instance.")

        # Get vertices of the polygon in flattened arrays
        poly_x = self.x.flatten()
        poly_y = self.y.flatten()

        # Repeat the polygon coordinates for each point
        n = len(poly_x)
        px = points.x.flatten()[:, np.newaxis]
        py = points.y.flatten()[:, np.newaxis]

        # Roll arrays for edge coordinates (xi, yi) -> (xj, yj) for each edge
        xi, yi = poly_x, poly_y
        xj, yj = np.roll(poly_x, -1), np.roll(poly_y, -1)

        # Check if the point is within the y-bounds of each polygon edge
        intersect = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        )

        # Count intersections for each point to determine if inside
        inside = intersect.sum(axis=1) % 2 == 1

        # Check if points are vertices
        if include_vertices:
            # Using np.isin to check for vertex containment
            points_tuples = [tuple(row) for row in points]
            polygon_tuples = [tuple(row) for row in polygon]

            # Use set intersection to find matching pairs
            matching_coordinates = np.array(
                list(set(points_tuples) & set(polygon_tuples))
            )
            inside |= vertex_contains

        # Return a Point instance with contained points, or boolean array
        return points[inside] if return_points else inside

    def contains_any(self, points):
        return any(self.contains(points, return_points=False))

    def contains_all(self, points):
        return all(self.contains(points, return_points=False))
