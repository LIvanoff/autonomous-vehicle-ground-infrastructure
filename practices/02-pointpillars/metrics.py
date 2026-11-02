"""Учебные precision и recall по сохранённым предсказаниям, без запуска сети."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def count_matches(boxes, scores, labels, gt_boxes, gt_labels):
    """Совпадение: тот же класс, расстояние между центрами в XY меньше 2 м."""
    used = set()
    tp = 0
    # Сначала рассматриваем самые уверенные предсказания.
    for index in np.argsort(-scores, kind="stable"):
        nearest = None
        best_distance = 2.0
        for target in range(len(gt_boxes)):
            if target in used or labels[index] != gt_labels[target]:
                continue
            distance = np.linalg.norm(boxes[index, :2] - gt_boxes[target, :2])
            if distance < best_distance:
                nearest = target
                best_distance = distance
        if nearest is not None:
            used.add(nearest)  # Одну рамку разметки нельзя засчитать дважды.
            tp += 1
    fp = len(boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, required=True, choices=range(1, 34))
    parser.add_argument("--frame", type=int, help="номер одного кадра; по умолчанию все пять")
    args = parser.parse_args()
    folder = ROOT / "output" / f"variant_{args.variant:02d}"
    if not (folder / "result.json").exists():
        parser.error("Нет предсказаний. Сначала выполните run.py для этого варианта.")
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    frames = [f for f in result["frames"] if args.frame is None or f["frame"] == args.frame]
    if not frames:
        parser.error("Этого кадра нет среди пяти кадров вашего варианта.")
    print(f"Вариант {args.variant}; кадры: {', '.join(str(f['frame']) for f in frames)}")
    print("Совпадение: одинаковый класс и расстояние XY < 2 м. Учебная оценка.")
    print("Порог    TP    FP    FN   Precision   Recall")
    for threshold in (0.2, 0.5, 0.8):
        tp = fp = fn = 0
        for frame in frames:
            with np.load(folder / frame["file"], allow_pickle=False) as data:
                keep = data["scores"] >= threshold
                counts = count_matches(data["boxes"][keep], data["scores"][keep],
                                       data["labels"][keep], data["gt_boxes"], data["gt_labels"])
                tp += counts[0]
                fp += counts[1]
                fn += counts[2]
        precision = f"{tp / (tp + fp):.3f}" if tp + fp else "—"
        recall = f"{tp / (tp + fn):.3f}" if tp + fn else "—"
        print(f"{threshold:5.1f} {tp:5d} {fp:5d} {fn:5d} {precision:>11} {recall:>8}")


if __name__ == "__main__":
    main()
