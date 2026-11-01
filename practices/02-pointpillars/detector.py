"""Готовый PointPillars. Подготовка pillars на NumPy не требует CUDA/spconv."""
from pathlib import Path
import urllib.request

import numpy as np
import torch
from torch_pointcloud.models import create_model
from torch_pointcloud.ops.box3d import nms3d

MODEL = "pointpillars-multihead.nuscenes.openpcdet"
WEIGHTS_URL = (
    "https://huggingface.co/torch-pointcloud/pointpillars-multihead.nuscenes.openpcdet/"
    "resolve/4de6b68e356558a9155b7452736d4641f198d7f6/model.safetensors"
)


def load_model():
    path = Path(__file__).resolve().parent / "weights/model.safetensors"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        print("Скачивание готовых весов PointPillars...", flush=True)
        temporary = path.with_suffix(".part")
        urllib.request.urlretrieve(WEIGHTS_URL, temporary)
        temporary.replace(path)
    return create_model(MODEL, task="detection", checkpoint_path=path).eval()


def make_pillars(points):
    """Первые 20 точек каждого pillar, максимум 30 000 pillars, как в конфигурации весов."""
    lower = np.array([-51.2, -51.2, -5.0], dtype=np.float32)
    upper = np.array([51.2, 51.2, 3.0], dtype=np.float32)
    size = np.array([0.2, 0.2, 8.0], dtype=np.float32)
    points = points[np.all((points[:, :3] >= lower) & (points[:, :3] < upper), axis=1)]
    grid = np.floor((points[:, :3] - lower) / size).astype(np.int32)
    voxels = np.zeros((30000, 20, 5), dtype=np.float32)
    coordinates = np.zeros((30000, 3), dtype=np.int32)
    counts = np.zeros(30000, dtype=np.int32)
    indices = {}
    for point, xyz in zip(points, grid):
        key = tuple(xyz[::-1])  # Модель ждёт индексы в порядке z, y, x.
        if key not in indices:
            if len(indices) == 30000:
                continue
            indices[key] = len(indices)
            coordinates[indices[key]] = key
        index = indices[key]
        if counts[index] < 20:
            voxels[index, counts[index]] = point
            counts[index] += 1
    count = len(indices)
    return (torch.from_numpy(voxels[:count]), torch.from_numpy(coordinates[:count]),
            torch.from_numpy(counts[:count]), torch.zeros(count, dtype=torch.long))


@torch.inference_mode()
def predict(model, points):
    raw = model(*make_pillars(points))
    detections = model.decode(raw)
    # Сохраняем все результаты от 0.1, чтобы затем сравнивать пороги 0.2/0.5/0.8.
    keep = detections["scores"] >= 0.1
    boxes = detections["boxes"][keep]
    scores = detections["scores"][keep]
    labels = detections["labels"][keep]
    # Постобработка из CPU-примера библиотеки: подавление дублей отдельно по классам.
    keep = nms3d(boxes, scores, iou_threshold=0.2, labels=labels)
    return boxes[keep].numpy(), scores[keep].numpy(), labels[keep].numpy()
