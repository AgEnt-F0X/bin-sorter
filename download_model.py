import urllib.request
from pathlib import Path

MODEL_URLS = [
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
    "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
    "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
]

MODEL_PATH = Path(__file__).resolve().parent / "models" / "mobilenetv2-12.onnx"
MIN_SIZE = 5 * 1024 * 1024   # the real model is ~13 MB; anything tiny is an HTML error page


def _looks_valid(path):
    return path.exists() and path.stat().st_size >= MIN_SIZE


def download_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _looks_valid(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return True
    if MODEL_PATH.exists():
        print(f"Existing model at {MODEL_PATH} is truncated/corrupt - re-downloading.")
        MODEL_PATH.unlink()

    part = MODEL_PATH.with_suffix(".part")
    for url in MODEL_URLS:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, part)
            if not _looks_valid(part):
                raise ValueError(f"download too small ({part.stat().st_size} bytes)")
            part.replace(MODEL_PATH)   # only a complete file ever gets the real name
            size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            print(f"Done! Saved to {MODEL_PATH} ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            print(f"  Failed: {e}")
            if part.exists():
                part.unlink()

    print("\nAll automatic downloads failed.")
    print("Manual download:")
    print("1. Go to: https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet")
    print("2. Download mobilenetv2-12.onnx")
    print(f"3. Save as: {MODEL_PATH}")
    print("\nOr use mobilenetv2-7.onnx from the same folder")
    return False


if __name__ == "__main__":
    raise SystemExit(0 if download_model() else 1)