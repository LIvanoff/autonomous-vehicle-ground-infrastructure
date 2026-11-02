"""Сопоставление рамок: дубли, классы, расстояние и пустые наборы."""
import unittest

import numpy as np

from metrics import count_matches


def boxes(*centers):
    return np.array([[x, y, 0, 4, 2, 1, 0] for x, y in centers]).reshape(-1, 7)


class MatchingTests(unittest.TestCase):
    def test_duplicate_is_false_positive(self):
        counts = count_matches(boxes((0, 0), (0.5, 0)), np.array([0.8, 0.9]),
                               np.array([0, 0]), boxes((0, 0)), np.array([0]))
        self.assertEqual(counts, (1, 1, 0))

    def test_wrong_class_and_distance_boundary(self):
        counts = count_matches(boxes((0, 0), (2, 0)), np.array([0.9, 0.8]),
                               np.array([1, 0]), boxes((0, 0)), np.array([0]))
        self.assertEqual(counts, (0, 2, 1))

    def test_confidence_order_and_nearest_unused_target(self):
        # Уверенная рамка x=1 сначала забирает ближайшую цель x=0.
        # Для x=0.1 оставшаяся цель x=2.5 уже слишком далеко.
        counts = count_matches(boxes((0.1, 0), (1, 0)), np.array([0.5, 0.9]),
                               np.array([0, 0]), boxes((0, 0), (2.5, 0)), np.array([0, 0]))
        self.assertEqual(counts, (1, 1, 1))

    def test_empty_sets(self):
        empty = np.array([])
        self.assertEqual(count_matches(boxes(), empty, empty, boxes((0, 0)), np.array([0])), (0, 0, 1))
        self.assertEqual(count_matches(boxes((0, 0)), np.array([0.8]), np.array([0]), boxes(), empty), (0, 1, 0))
        self.assertEqual(count_matches(boxes(), empty, empty, boxes(), empty), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
