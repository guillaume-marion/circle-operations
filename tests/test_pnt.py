import unittest

import numpy as np

from geom import Point, Polygon


class TestPoint(unittest.TestCase):

    def test_manual_creation(self):
        p = Point([6, 12])
        self.assertEqual(p.x, 6)
        self.assertEqual(p.y, 12)
        self.assertEqual(p.xy.tolist(), [[6, 12]])

    def test_random_creation(self):
        mp = Point.random(4)
        self.assertEqual(len(mp), 4)

    def test_drop(self):
        mp = Point.random(4)
        values = mp.tolist()
        mp = mp.drop(0)
        self.assertEqual(mp.tolist(), values[1:])

    def test_distance(self):
        sp1 = Point([3, 3])
        sp2 = Point([6, 6])
        mp1 = Point([[3, 3], [6, 6]])
        mp2 = Point([[3, 3], [6, 6]])
        self.assertEqual(sp1.distance(sp2).round(4), 4.2426)
        self.assertEqual(sp1.distance(mp1).round(4).tolist(), [[0.0], [4.2426]])
        self.assertEqual(mp1.distance(sp1).round(4).tolist(), [[0.0], [4.2426]])
        with self.assertRaises(AssertionError):
            mp1.distance(mp2)

    def test_angle_to_align(self):
        sp1 = Point([3, 3])
        sp2 = Point([6, 6])
        mp1 = Point([[3, 3], [6, 6]])
        mp2 = Point([[3, 3], [6, 6]])
        self.assertEqual(sp1.angle_to_align(sp2), 45)
        self.assertEqual(sp1.angle_to_align(mp1).tolist(), [[0], [45]])
        self.assertEqual(mp1.angle_to_align(sp1).tolist(), [[0], [45]])
        with self.assertRaises(AssertionError):
            mp1.angle_to_align(mp2)

    def test_angle_between(self):
        p0 = Point([10, 0])
        p1 = Point([5, 0])
        p2 = Point([5, 5])
        mp1 = Point([[5, 0], [5, 5]])
        mp2 = Point([[10, 0], [10, 2.5]])
        self.assertEqual(p0.angle_between(p1, p2), [45.0])
        self.assertEqual(p0.angle_between(p1, mp1).tolist(), [[0], [45.0]])
        self.assertEqual(
            mp2.angle_between(p1, p2).round(4).tolist(), [[45.0], [53.1301]]
        )
        with self.assertRaises(AssertionError):
            p0.angle_between(mp1, mp2)
        with self.assertRaises(AssertionError):
            mp2.angle_between(p1, mp1)

    def test_centroid(self):
        p = Point([3, 3])
        mp = Point([[3, 0], [6, 10]])
        self.assertEqual(p.centroid().tolist(), [[3, 3]])
        self.assertEqual(mp.centroid().tolist(), [[4.5, 5.0]])

    def test_ordering(self):
        mp = Point([[0, 0], [0, 5], [5, 0], [5, 5]])
        center = Point([2.5, 10])
        start = Point([2.5, 0])
        start2 = Point([0, 2.5])
        self.assertEqual(
            mp.order_clockwise().tolist(), [[0, 5], [5, 5], [5, 0], [0, 0]]
        )
        self.assertEqual(
            mp.order_clockwise(center=center).tolist(), [[5, 5], [5, 0], [0, 0], [0, 5]]
        )
        self.assertEqual(
            mp.order_clockwise(start=start).tolist(), [[0, 0], [0, 5], [5, 5], [5, 0]]
        )
        self.assertEqual(
            mp.order_clockwise(start=start2, center=center).tolist(),
            [[0, 5], [5, 5], [5, 0], [0, 0]],
        )


class TestPolygon(unittest.TestCase):

    def test_manual_creation(self):
        sp = Point([0, 0])
        mp1 = Point([[0, 0], [0, 5]])
        mp2 = Point([[0, 0], [5, 5], [0, 5], [9, 9]])
        po1 = Polygon(mp2)
        with self.assertRaises(ValueError):
            Polygon(sp)
        with self.assertRaises(ValueError):
            Polygon(mp1)
        self.assertNotEqual(mp2.tolist(), po1.tolist())
        self.assertEqual(mp2.order_clockwise().tolist(), po1.tolist())

    def test_random_creation(self):
        with self.assertRaises(ValueError):
            Polygon.random(0)
            Polygon.random(1)
            Polygon.random(2)
        po1 = Polygon.random(4)
        self.assertEqual(po1.count, 4)

    def test_getitem(self):
        po = Polygon.random(5)
        self.assertEqual(type(po), Polygon)
        self.assertEqual(type(po[0]), Point)
        self.assertEqual(type(po[:2]), Point)
        self.assertEqual(type(po[3:]), Point)
        self.assertEqual(type(po[2:4]), Point)
        self.assertEqual(type(po[1:]), Polygon)
        self.assertEqual(type(po[:4]), Polygon)
        self.assertEqual(type(po[1:5]), Polygon)

    def test_area(self):
        po1 = Polygon([[0, 0], [5, 5], [0, 5], [9, 9]])
        po2 = Polygon([[0, 0], [0, 5], [5, 5], [5, 0]])
        self.assertEqual(po1.area(), 22.5)
        self.assertEqual(po2.area(), 25.0)

    # def test_contains(self):
    #     coords = [[0, 0], [5, 5], [0, 5], [9, 9]]
    #     po1 = Polygon(coords)
    #     coords.extend([[10, 10], [0, 2.5], [2.5, 5], [5, 6], [9, 8]])
    #     mp2 = Point(coords)
    #     print(po1.contains(mp2))
    #     # self.assertEqual(po1.contains(mp2))
    #     pass


if __name__ == "__main__":

    coords = [[0, 0], [5, 5], [0, 5], [9, 9]]
    po1 = Polygon(coords)
    coords2 = coords + [[10, 10], [0, 2.5], [8, 8], [5, 0]]
    mp2 = Point(coords2)
    # print(po1)
    # print(mp2.order_clockwise())
    # print(po1.contains(mp2, return_points=False, include_vertices=False))
    print(mp2, "\n")
    print(po1, "\n")
    print(...)

    unittest.main()
