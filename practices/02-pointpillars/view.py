"""Показать сохранённые предсказания в Rerun; нейросеть повторно не запускается."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import rerun as rr

ROOT = Path(__file__).resolve().parent


def draw_boxes(recording, path, boxes, labels, classes, color, scores=None):
    # Rerun принимает кватернионы xyzw; у нас поворот yaw вокруг оси z.
    quaternions = np.zeros((len(boxes), 4), dtype=np.float32)
    quaternions[:, 2] = np.sin(boxes[:, 6] / 2)
    quaternions[:, 3] = np.cos(boxes[:, 6] / 2)
    captions = [classes[int(label)] for label in labels]
    if scores is not None:
        captions = [f"{name} {score:.2f}" for name, score in zip(captions, scores)]
    recording.log(path, rr.Boxes3D(centers=boxes[:, :3], half_sizes=boxes[:, 3:6] / 2,
                                 quaternions=quaternions, colors=color,
                                 labels=captions, show_labels=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, required=True, choices=range(1, 34))
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--save-only", action="store_true", help="сохранить .rrd без открытия окна")
    args = parser.parse_args()
    folder = ROOT / "output" / f"variant_{args.variant:02d}"
    if not (folder / "result.json").exists():
        parser.error("Нет предсказаний. Сначала выполните run.py для этого варианта.")
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if not result["min_score"] <= args.score <= 1:
        parser.error(f"Порог должен быть от {result['min_score']} до 1.")
    path = folder / f"score_{args.score:g}.rrd"
    with rr.RecordingStream("practice_02_pointpillars") as recording:
        recording.save(str(path))
        recording.log("lidar", rr.ViewCoordinates.RFU, static=True)
        recording.log("info", rr.TextDocument(
            f"Variant {args.variant}; {result['scene']}; score >= {args.score}\n"
            "Orange: PointPillars. Green: ground truth. Points: current LiDAR scan.\n"
            "Coordinates: current LiDAR frame, metres. Boxes are not tracks."
        ), static=True)
        for frame in result["frames"]:
            recording.set_time("frame", sequence=frame["frame"])
            with np.load(folder / frame["file"], allow_pickle=False) as data:
                keep = data["scores"] >= args.score
                recording.log("lidar/points", rr.Points3D(data["points"], colors=[170, 180, 190], radii=0.035))
                draw_boxes(recording, "lidar/ground_truth", data["gt_boxes"], data["gt_labels"],
                           result["classes"], [60, 210, 100])
                draw_boxes(recording, "lidar/predictions", data["boxes"][keep], data["labels"][keep],
                           result["classes"], [255, 150, 40], data["scores"][keep])
                print(f"Кадр {frame['frame']}: {int(keep.sum())} обнаружений при пороге {args.score}")
    print("Запись:", path)
    if not args.save_only:
        subprocess.run([sys.executable, "-m", "rerun", str(path)], check=True)


if __name__ == "__main__":
    main()
