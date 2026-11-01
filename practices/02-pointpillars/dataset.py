"""Чтение выбранных кадров nuScenes-mini. Все координаты — в системе лидара."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from torch_pointcloud.datasets.nuscenes import load_nuscenes_sweeps

CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
CATEGORIES = {
    "vehicle.car": "car", "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle", "vehicle.trailer": "trailer",
    "vehicle.bus.bendy": "bus", "vehicle.bus.rigid": "bus",
    "vehicle.motorcycle": "motorcycle", "vehicle.bicycle": "bicycle",
    "movable_object.barrier": "barrier", "movable_object.trafficcone": "traffic_cone",
    "human.pedestrian.adult": "pedestrian", "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian", "human.pedestrian.police_officer": "pedestrian",
}


def read_tables(data_dir):
    """Индекс token → запись позволяет быстро находить связанные записи JSON."""
    tables = {}
    for name in ("scene", "sample", "sample_data", "ego_pose", "calibrated_sensor",
                 "sensor", "sample_annotation", "instance", "category"):
        path = data_dir / "v1.0-mini" / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Нет {path}. Сначала выполните python download_data.py")
        records = json.loads(path.read_text(encoding="utf-8"))
        tables[name] = {record["token"]: record for record in records}
    return tables


def scene_frames(tables, scene_name):
    scene = next(s for s in tables["scene"].values() if s["name"] == scene_name)
    frames = []
    for record in tables["sample_data"].values():
        sample = tables["sample"][record["sample_token"]]
        calibration = tables["calibrated_sensor"][record["calibrated_sensor_token"]]
        sensor = tables["sensor"][calibration["sensor_token"]]
        if sample["scene_token"] == scene["token"] and record["is_key_frame"] and sensor["channel"] == "LIDAR_TOP":
            frames.append(record)
    return sorted(frames, key=lambda frame: frame["timestamp"])


def pose(record):
    """Матрица перехода из локальной системы координат в родительскую."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(record["rotation"], scalar_first=True).as_matrix()
    matrix[:3, 3] = record["translation"]
    return matrix


def ground_truth(tables, frame):
    ego = tables["ego_pose"][frame["ego_pose_token"]]
    calibration = tables["calibrated_sensor"][frame["calibrated_sensor_token"]]
    lidar_from_global = np.linalg.inv(pose(ego) @ pose(calibration))
    boxes, labels = [], []
    for annotation in tables["sample_annotation"].values():
        if annotation["sample_token"] != frame["sample_token"]:
            continue
        instance = tables["instance"][annotation["instance_token"]]
        category = tables["category"][instance["category_token"]]["name"]
        name = CATEGORIES.get(category)
        if name is None or annotation["num_lidar_pts"] == 0:
            continue
        transform = lidar_from_global @ pose(annotation)
        x, y, z = transform[:3, 3]
        if not (-51.2 <= x < 51.2 and -51.2 <= y < 51.2):
            continue
        width, length, height = annotation["size"]
        yaw = np.arctan2(transform[1, 0], transform[0, 0])
        boxes.append([x, y, z, length, width, height, yaw])
        labels.append(CLASSES.index(name))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 7), np.asarray(labels, dtype=np.int64)


def load_frame(data_dir, tables, frame):
    # Модель обучена на ключевом скане и предыдущих сканах (до 10 суммарно).
    points = load_nuscenes_sweeps(data_dir, frame, tables["ego_pose"],
                                 tables["calibrated_sensor"], tables["sample_data"], max_sweeps=10)
    current = np.fromfile(Path(data_dir) / frame["filename"], dtype=np.float32).reshape(-1, 5)
    boxes, labels = ground_truth(tables, frame)
    return points, current[:, :3], boxes, labels
