"""Скачать nuScenes-mini и извлечь только лидар, разметку и калибровки."""
import argparse
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="использовать уже скачанный v1.0-mini.tgz")
    args = parser.parse_args()
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    archive = args.archive or data / "v1.0-mini.tgz"
    if args.archive and not archive.is_file():
        parser.error(f"Архив не найден: {archive}")
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        print("Загрузка официального nuScenes-mini (около 4 ГБ)...", flush=True)
        with urllib.request.urlopen(URL, timeout=60) as source, temporary.open("wb") as target:
            expected_size = int(source.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := source.read(8 * 1024 * 1024):
                target.write(chunk)
                downloaded += len(chunk)
                if downloaded % (128 * 1024 * 1024) == 0:
                    print(f"  {downloaded / 1024**3:.1f} ГБ", flush=True)
        if expected_size and downloaded != expected_size:
            raise RuntimeError("Архив скачан не полностью. Повторите загрузку.")
        temporary.replace(archive)
    print("Извлечение лидарных данных и метаданных...", flush=True)
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            parts = Path(member.name).parts
            metadata = len(parts) == 2 and parts[0] == "v1.0-mini" and parts[1].endswith(".json")
            lidar = len(parts) == 3 and parts[0] in ("samples", "sweeps") and parts[1] == "LIDAR_TOP"
            if member.isfile() and (metadata or lidar):
                source.extract(member, path=data, filter="data")
    if not (data / "v1.0-mini/scene.json").is_file():
        raise RuntimeError("В архиве не найдены метаданные v1.0-mini")
    print("Данные готовы:", data)


if __name__ == "__main__":
    main()
