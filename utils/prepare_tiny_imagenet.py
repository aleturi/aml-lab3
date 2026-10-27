from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (
    PROJECT_ROOT / "downloads" / "tiny-imagenet" / "tiny-imagenet-200"
)
VAL_DIR = DATASET_ROOT / "val"
ANNOTATIONS_FILE = VAL_DIR / "val_annotations.txt"
IMAGES_DIR = VAL_DIR / "images"


def main() -> None:
    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Validation annotations not found: {ANNOTATIONS_FILE}"
        )

    if not IMAGES_DIR.exists():
        print("Validation images are already organized.")
        return

    with ANNOTATIONS_FILE.open(encoding="utf-8") as annotations:
        for line in annotations:
            filename, class_name, *_ = line.rstrip("\n").split("\t")
            class_dir = VAL_DIR / class_name
            class_dir.mkdir(exist_ok=True)
            shutil.copyfile(IMAGES_DIR / filename, class_dir / filename)

    shutil.rmtree(IMAGES_DIR)
    print(f"Validation set prepared in {VAL_DIR}")


if __name__ == "__main__":
    main()