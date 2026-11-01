"""Обработать пять кадров своего варианта и сохранить предсказания на CPU."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from dataset import CLASSES, load_frame, read_tables, scene_frames
from detector import MODEL, load_model, predict

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, required=True, choices=range(1, 34))
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    variant = json.loads((ROOT / "variants.json").read_text(encoding="utf-8"))[str(args.variant)]
    tables = read_tables(args.data)
    frames = scene_frames(tables, variant["scene"])
    torch.set_num_threads(4)
    model = load_model()
    output = ROOT / "output" / f"variant_{args.variant:02d}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").unlink(missing_ok=True)  # Незавершённый запуск нельзя открыть как готовый.
    records = []
    for number in variant["frames"]:
        frame = frames[number - 1]  # В README кадры нумеруются с единицы.
        print(f"{variant['scene']}, кадр {number}: обработка...", flush=True)
        started = time.perf_counter()
        points, current, gt_boxes, gt_labels = load_frame(args.data, tables, frame)
        boxes, scores, labels = predict(model, points)
        seconds = time.perf_counter() - started
        filename = f"frame_{number:02d}.npz"
        np.savez_compressed(output / filename, points=current, boxes=boxes, scores=scores,
                            labels=labels, gt_boxes=gt_boxes, gt_labels=gt_labels)
        records.append({"frame": number, "sample_token": frame["sample_token"],
                        "timestamp": frame["timestamp"], "file": filename, "seconds": round(seconds, 3)})
        print(f"  {len(boxes)} обнаружений от 0.1; {seconds:.1f} с", flush=True)
    manifest = {"variant": args.variant, "scene": variant["scene"], "model": MODEL,
                "classes": CLASSES, "min_score": 0.1, "frames": records}
    (output / "result.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово. Просмотр: python view.py --variant {args.variant}")


if __name__ == "__main__":
    main()
