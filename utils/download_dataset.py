from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from prepare_tiny_imagenet import main as prepare_validation_set


DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
ARCHIVE_PATH = DOWNLOADS_DIR / "tiny-imagenet-200.zip"
EXTRACT_DIR = DOWNLOADS_DIR / "tiny-imagenet"
DATASET_DIR = EXTRACT_DIR / "tiny-imagenet-200"


def main() -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    if not ARCHIVE_PATH.exists():
        print(f"Downloading Tiny ImageNet to {ARCHIVE_PATH}")
        urlretrieve(DATASET_URL, ARCHIVE_PATH)

    if not DATASET_DIR.exists():
        print(f"Extracting Tiny ImageNet to {EXTRACT_DIR}")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with ZipFile(ARCHIVE_PATH) as archive:
            archive.extractall(EXTRACT_DIR)

    prepare_validation_set()
    print(f"Dataset available at {DATASET_DIR}")


if __name__ == "__main__":
    main()
