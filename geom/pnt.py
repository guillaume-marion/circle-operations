"""
This module provides a custom Point class that subclasses np.ndarray.
The Point class is designed to represent one or multiple points in a 2D space.

The Point class leverages vectorized operations from np.ndarray and provides
additional functionality for working with points.
"""

from typing import Optional, Union

import numpy as np

from geom.utils import atan2_v, degrees_v


class Point(np.ndarray):
    """
    The Point class subclasses np.ndarray to represent one or multiple points in 2D space.

    A Point can be created from a np.ndarray with shape (2,) or (*,2), where each row
    represents a point in 2D space.

    Attributes:
        inputarray (np.ndarray): The input array used to create the Point instance.

    Properties:
        x: Access or set the x-coordinates of the point(s).
        y: Access or set the y-coordinates of the point(s).
        xy: Access or set the x,y-coordinates of the point(s).
        count: The number of points represented by the Point instance.

    Methods:
        distance: Compute the distance between points.
        angle_to_align: Compute the angle to align points.
        angle_between: Compute the angle between points.
        centroid: Compute the centroid of points.
        order_clockwise: Order points clockwise.

    Examples:
        >>> p = Point([5,7])
        >>> mp = Point([[5,7],[13,4]])
    """

    def __new__(cls, inputarray):
        """
        Creates a new Point instance from a np.ndarray.

        Args:
            inputarray (np.ndarray): The input array with shape (2,) or (*,2).

        Returns:
            Point: A new Point instance.

        Raises:
            ValueError: If the input array does not have the correct shape.
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
        Initializes the Point instance.

        Note:
            This method is not intended to be called directly. Instead, use the
            __new__ method to create a new Point instance. This is related to the
            proper subclassing of np.ndarray.
        """
        pass

    @property
    def x(self) -> np.ndarray:
        """
        The x-coordinates of the point(s).

        Returns:
            np.ndarray: The x-coordinates.
        """
        return np.asarray(self[:, :1])

    @x.setter
    def x(self, value: np.ndarray):
        """
        Sets the x-coordinates of the point(s).

        Args:
            value (np.ndarray): The new x-coordinates.
        """
        self[:, :1] = value

    @property
    def y(self) -> np.ndarray:
        """
        The y-coordinates of the point(s).

        Returns:
            np.ndarray: The y-coordinates.
        """
        return np.asarray(self[:, 1:2])

    @y.setter
    def y(self, value: np.ndarray):
        """
        Sets the y-coordinates of the point(s).

        Args:
            value (np.ndarray): The new y-coordinates.
        """
        self[:, 1:2] = value

    @property
    def xy(self) -> np.ndarray:
        """
        The x,y-coordinates of the point(s).

        Returns:
            np.ndarray: The x,y-coordinates.
        """
        return self[:, :]

    @xy.setter
    def xy(self, value: np.ndarray):
        """
        Sets the x,y-coordinates of the point(s).

        Args:
            value (np.ndarray): The new x,y-coordinates.
        """
        self[:, :] = value

    @property
    def count(self) -> int:
        """
        The number of point(s).

        Returns:
            int: The number of points.
        """
        return self.shape[0]

    def __getitem__(
        self, val: Union[int, slice, np.ndarray]
    ) -> Union["Point", np.ndarray]:
        """
        Returns a new Point instance or a numpy.ndarray based on the input value.

        If val is an integer, it is treated as a row selection, for which we need to force a Point instance creation.
        If val is a slice or numpy.ndarray, numpy's __getitem__ method is used, resulting in a Point instance by default.

        Usecase: Selecting a single row from a multiple Point instance should
        also result in a Point with the correct shape (i.e. double enclosing brackets)
        to allow further use of the Point class' methods.

        Args:
            val (Union[int, slice, np.ndarray]): The input value for indexing.

        Returns:
            Union['Point', np.ndarray]: A new Point instance or a numpy.ndarray.
        """
        if isinstance(val, int):
            return self.__class__(np.asarray(self)[val])
        else:
            return super(Point, self).__getitem__(val)

    def __repr__(self):
        """
        Returns a string representation of the Point instance.

        The string shows the class name and the coordinates in a formatted manner.
        """
        return str(self.__class__.__name__) + "([\n {:>10}\n])".format(
            self.__str__()[1:-1]
        )

    def __round__(self, decimals=0):
        """
        Defines custom __round__ method, which we can use to return a class
            compared to an array when using .round() method.
        """
        rounded_array = self.round(decimals)
        rounded_class = self.__class__(rounded_array)
        return rounded_class

    @staticmethod
    def _random(
        size: int, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> np.ndarray:
        """
        Generates a random numpy.ndarray of shape (size, 2).

        Args:
            size (int): The number of rows in the output array.
            x_min (float): The minimum value for the x-coordinates.
            x_max (float): The maximum value for the x-coordinates.
            y_min (float): The minimum value for the y-coordinates.
            y_max (float): The maximum value for the y-coordinates.

        Returns:
            np.ndarray: A numpy.ndarray with shape (size, 2) containing the random points.
        """
        x = np.random.uniform(x_min, x_max, size)
        y = np.random.uniform(y_min, y_max, size)
        xy = np.dstack([x, y])
        return xy

    @classmethod
    def random(
        cls,
        size: int,
        x_min: float = 0,
        x_max: float = 10,
        y_min: float = 0,
        y_max: float = 10,
    ) -> "Point":
        """
        Generates a random set of points.

        Args:
            size (int): The number of rows in the output array.
            x_min (float): The minimum value for the x-coordinates.
            x_max (float): The maximum value for the x-coordinates.
            y_min (float): The minimum value for the y-coordinates.
            y_max (float): The maximum value for the y-coordinates.

        Returns:
            Point: A Point instance based on a random set of points.
        """
        xy_values = cls._random(
            size=size, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
        random_point = cls(xy_values)
        return random_point

    def drop(self, row: int) -> "Point":
        """
        Returns a new Point instance without the specified row.

        Args:
            row (int): The index of the row to be dropped.

        Returns:
            Point: The Point without the specified row.
        """
        lower_end = self[:row]
        upper_end = self[row + 1 :]
        appended = np.append(lower_end, upper_end, axis=0)
        appended_point = self.__class__(appended)
        return appended_point

    def dropna(self) -> "Point":
        """
        Returns a new Point instance without nan values.

        Returns:
            Point: The Point without nan values.
        """
        mask = np.array((np.isnan(self) == False).any(axis=1))
        self_without_nan = self[mask, :]
        return self_without_nan

    @staticmethod
    def _distance(
        px: Union[int, float, np.ndarray],
        py: Union[int, float, np.ndarray],
        mpx: np.ndarray,
        mpy: np.ndarray,
    ) -> Union[float, np.ndarray]:
        """
        Calculate the Euclidean distance(s) between points.

        Args:
            px (Union[int, float, np.ndarray]): The x-coordinate(s) of the reference point. This can be a scalar or an array.
            py (Union[int, float, np.ndarray]): The y-coordinate(s) of the reference point. This can be a scalar or an array.
            mpx (np.ndarray): The x-coordinate(s) of the point(s) to calculate the distance to. This is an array.
            mpy (np.ndarray): The y-coordinate(s) of the point(s) to calculate the distance to. This is an array.

        Returns:
            Union[float, np.ndarray]: The Euclidean distance(s) between the reference point and the point(s) specified by [mpx, mpy].
        """
        return np.sqrt((mpx - px) ** 2 + (mpy - py) ** 2)

    def distance(self, other_points: "Point") -> Union[float, np.ndarray]:
        """
        Calculate the Euclidean distance(s) between points.

        Args:
            other_point (Point): another Point instance representing one (if self represents multiple points) or multiple points.

        Returns:
            Union[float, np.ndarray]: The euclidean distance(s) between self and the other point(s).
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
    def _angle_to_align(
        px: Union[int, float, np.ndarray],
        py: Union[int, float, np.ndarray],
        mpx: np.ndarray,
        mpy: np.ndarray,
    ) -> Union[float, np.ndarray]:
        """
        Compute the angle(s) needed to align one point with other points.

        Args:
            px (Union[int, float, np.ndarray]): One x-coordinate as a scalar or array.
            py (Union[int, float, np.ndarray]): One y-coordinate as a scalar or array.
            mpx (np.ndarray): One or multiple x-coordinates as an array.
            mpy (np.ndarray): One or multiple y-coordinates as an array.

        Returns:
            Union[float, np.ndarray]: The angle(s) at which (a) fully horizontal line(s) needs to rotate
            clockwise in order to match the line(s) between the point defined by [px, py] and the point(s) defined by [mpx, mpy].
        """
        dx = mpx - px
        dy = mpy - py
        return degrees_v(atan2_v(dy, dx))

    def angle_to_align(self, other_points: "Point") -> Union[float, np.ndarray]:
        """
        Compute the angle(s) needed to align one point with other points.

        Args:
            other_point (Point): another Point instance representing one (if self represents multiple points) or multiple points.

        Returns:
            Union[float, np.ndarray]: The angle(s), at which the horizontal line(s)
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

    def angle_between(self, point: "Point", other_points: "Point") -> np.ndarray:
        """
        Compute the angles between a given point and other points with self as the vertex.

        Args:
            point (Point): A single point from which the clockwise angle to the other point(s) is calculated.
            other_points (Point): One or multiple points.

        Returns:
            np.ndarray: The angles between the point and the other
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
    def _centroid(mpx: np.ndarray, mpy: np.ndarray) -> np.ndarray:
        """
        Compute the centroid of a set of points.

        Args:
            mpx (np.ndarray): One or multiple x-coordinates as an array.
            mpy (np.ndarray): One or multiple y-coordinates as an array.

        Returns:
            np.ndarray: The centroid.
        """
        centroid = [mpx.sum() / mpx.size, mpy.sum() / mpx.size]
        return centroid

    def centroid(self) -> "Point":
        """
        Compute the centroid of a set of points.

        Returns:
            Point: The centroid of self as a Point.
        """
        centroid = self._centroid(mpx=self.x, mpy=self.y)
        return Point(centroid)

    def order_clockwise(
        self,
        start: Optional["Point"] = None,
        center: Optional["Point"] = None,
        return_angles: bool = False,
    ) -> "Point":
        """
        Order the points in a clockwise fashion.

        Args:
            start (Optional[Point]): The starting point for computing the clockwise order. Defaults to the horizontal line as starting point.
            center (Optional[Point]): The center for the clock-pointer. Defaults to the centroid of self.
            return_angles (bool): Whether or not to return the angles used to order the points.

        Returns:
            Point: The Points in a clockwise-ordered fashion.
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

    A Polygon can be created from a np.ndarray with shape (*,2), where each row
    represents a point in 2D space, or a Point consisting of at least 3 points.
    The points are ordered in a clockwise fashion at initialization.

    Attributes:
        inputarray (Union[np.ndarray, Point]): The input array used to create the Polygon instance.

    Examples:
        >>> p = Polygon([[5,7],[13,4],[7,7]])
    """

    def __new__(cls, inputarray):
        """
        Creates a new Polygon instance from a np.ndarray.

        Args:
            inputarray (np.ndarray): The input array with shape (*,2).

        Returns:
            Polygon: A new Polygon instance.

        Methods:
            area: Calculate the area of the polygon.
            contains: Check if points are inside the polygon.

        Raises:
            ValueError: If the input array does not have the correct shape or if it has fewer than 3 points.
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
        Initializes the Polygon instance.

        Note:
            This method is not intended to be called directly. Instead, use the
            __new__ method to create a new Polygon instance. This is related to the
            proper subclassing of np.ndarray.
        """
        pass

    def __getitem__(self, val):
        """
        If val is an int or results in fewer than 3 points, return a Point instance.
        """
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
    def _area(mpx: np.ndarray, mpy: np.ndarray) -> float:
        """
        Compute the area of a polygon.

        Args:
            mpx (np.ndarray): Multiple x-coordinates in clockwise order.
            mpy (np.ndarray): Multiple y-coordinates in clockwise order.

        Returns:
            float: The area of the polygon bounded by the points.
        """
        return 0.5 * np.abs(
            np.dot(mpx.T[0], np.roll(mpy.T[0], 1))
            - np.dot(mpy.T[0], np.roll(mpx.T[0], 1))
        )

    def area(self) -> float:
        """
        Compute the area of the polygon.

        Returns:
            float: The area of the polygon.
        """
        area = self._area(self.x, self.y).item()
        return area

    def contains(
        self, points: "Point", return_points: bool = True, include_vertices: bool = True
    ) -> Union["Point", np.ndarray]:
        """
        Check if points are inside the polygon.

        Args:
            points (Point): A Point instance with one or multiple points to check for containment.
            return_points (bool): Whether to return the points that are contained or the boolean array indicating which points are inside.
            include_vertices (bool): Whether or not to consider vertices as contained.

        Returns:
            Union[Point, np.ndarray]: A Point instance containing only those points within the polygon, or a boolean array indicating which points are inside.
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

    def contains_any(self, points: "Point") -> bool:
        """
        Check if any of the points are inside the polygon.

        Args:
            points (Point): A Point instance with the points to check.

        Returns:
            bool: True if any of the points are inside the polygon, False otherwise.
        """
        return any(self.contains(points, return_points=False))

    def contains_all(self, points: "Point") -> bool:
        """
        Check if all of the points are inside the polygon.

        Args:
            points (Point): A Point instance with the points to check.

        Returns:
            bool: True if all of the points are inside the polygon, False otherwise.
        """
        return all(self.contains(points, return_points=False))
