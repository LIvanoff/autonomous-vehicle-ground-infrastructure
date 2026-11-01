"""Проверки подготовки данных. Запуск: python -m unittest discover -s tests -v."""
import unittest

import numpy as np

from dataset import ground_truth
from detector import make_pillars


class PillarTests(unittest.TestCase):
    def test_grouping_range_and_padding(self):
        points = np.array([
            [0.05, 0.05, 0, 0.2, 0], [0.10, 0.10, 0, 0.7, 0.05],
            [0.25, 0.05, 0, 0.8, 0], [51.2, 0, 0, 1, 0], [-51.3, 0, 0, 1, 0],
        ], dtype=np.float32)
        voxels, coordinates, counts, batch = make_pillars(points)
        self.assertEqual(counts.tolist(), [2, 1])
        self.assertEqual(coordinates.tolist(), [[0, 256, 256], [0, 256, 257]])
        np.testing.assert_array_equal(voxels[0, :2].numpy(), points[:2])
        self.assertEqual(int(voxels[0, 2:].count_nonzero()), 0)
        self.assertEqual(batch.tolist(), [0, 0])

    def test_keeps_first_twenty_points(self):
        points = np.zeros((25, 5), dtype=np.float32)
        points[:, 3] = np.arange(25)
        voxels, _, counts, _ = make_pillars(points)
        self.assertEqual(counts.tolist(), [20])
        np.testing.assert_array_equal(voxels[0, :, 3].numpy(), np.arange(20))


class CoordinateTests(unittest.TestCase):
    def test_global_annotation_becomes_lidar_box(self):
        tables = {
            "ego_pose": {"ego": {"translation": [10, 0, 0], "rotation": [1, 0, 0, 0]}},
            "calibrated_sensor": {"calib": {"translation": [1, 0, 0], "rotation": [1, 0, 0, 0]}},
            "instance": {"object": {"category_token": "car"}},
            "category": {"car": {"name": "vehicle.car"}},
            "sample_annotation": {"box": {
                "sample_token": "sample", "instance_token": "object", "num_lidar_pts": 10,
                "translation": [13, 2, 0], "rotation": [1, 0, 0, 0], "size": [2, 4, 1],
            }},
        }
        frame = {"sample_token": "sample", "ego_pose_token": "ego", "calibrated_sensor_token": "calib"}
        boxes, labels = ground_truth(tables, frame)
        np.testing.assert_allclose(boxes, [[2, 2, 0, 4, 2, 1, 0]])
        self.assertEqual(labels.tolist(), [0])
        # Поворот лидара на 90 градусов переводит (2, 2) в (2, -2).
        tables["calibrated_sensor"]["calib"]["rotation"] = [2**-0.5, 0, 0, 2**-0.5]
        boxes, _ = ground_truth(tables, frame)
        np.testing.assert_allclose(boxes, [[2, -2, 0, 4, 2, 1, -np.pi / 2]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
