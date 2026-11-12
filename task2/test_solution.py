"""Проверки ПЗ №2, вариант 11. Запуск из корня: python -m unittest discover -s task2."""
import math
import unittest
from solution import solve_cauchy, solve_boundary, boundary_matrix


class SolutionTests(unittest.TestCase):
    def test_cauchy(self):
        rows = solve_cauchy()
        self.assertEqual(len(rows), 11)
        self.assertAlmostEqual(rows[0]["x"], 1)
        self.assertAlmostEqual(rows[-1]["x"], 3)
        for name in ("euler", "euler_cauchy", "rk4"):
            self.assertAlmostEqual(rows[0][name], 1.5)
            self.assertTrue(all(math.isfinite(row[name]) for row in rows))
        self.assertAlmostEqual(rows[-1]["rk4"], 20.8033625757, places=6)

    def test_boundary(self):
        rows, residual = solve_boundary()
        ys = [row["y"] for row in rows]
        self.assertEqual(len(ys), 4)
        self.assertAlmostEqual((ys[1]-ys[0])/0.1, 1.6, places=9)
        self.assertAlmostEqual(2*ys[3]+3*(ys[3]-ys[2])/0.1, 4.8, places=9)
        self.assertLess(residual, 1e-11)
        self.assertAlmostEqual(ys[-1], 6.4156363636, places=7)

    def test_matrix(self):
        xs, A, b = boundary_matrix()
        self.assertEqual((len(xs), len(A), len(b)), (4,4,4))
        self.assertTrue(all(len(row) == 4 for row in A))


if __name__ == "__main__":
    unittest.main()
