import sys
from pathlib import Path

BASE = Path(__file__).parent
print("BinSorter - Setup Verification")
print("=" * 40)

checks = []

# Check Python version
checks.append(("Python version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", sys.version_info >= (3, 8)))

# Check required files
files = [
    ("requirements.txt", BASE / "requirements.txt"),
    ("src/app.py", BASE / "src" / "app.py"),
    ("download_model.py", BASE / "download_model.py"),
    ("data/imagenet_labels.json", BASE / "data" / "imagenet_labels.json"),
    ("data/label_to_bin.json", BASE / "data" / "label_to_bin.json"),
    ("data/bins.json", BASE / "data" / "bins.json"),
    ("data/fallback_items.json", BASE / "data" / "fallback_items.json"),
    ("data/counter.json", BASE / "data" / "counter.json"),
]

for name, path in files:
    exists = path.exists()
    checks.append((name, "Found" if exists else "MISSING", exists))

# Check model
model_path = BASE / "models" / "mobilenetv2-12.onnx"
model_exists = model_path.exists()
checks.append(("Model (mobilenetv2-12.onnx)", f"{model_path.stat().st_size/1024/1024:.1f} MB" if model_exists else "MISSING - run download_model.py", model_exists))

# Check imports
imports = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("onnxruntime", "onnxruntime"),
    ("PIL", "Pillow"),
    ("tkinter", "tkinter (reinstall Python with Tcl/Tk) -"),
]

for mod, pkg in imports:
    try:
        __import__(mod)
        checks.append((f"Import {mod}", "OK", True))
    except ImportError:
        checks.append((f"Import {mod}", f"MISSING - pip install {pkg}", False))

print()
for name, status, ok in checks:
    icon = "[OK]" if ok else "[FAIL]"
    print(f"  {icon} {name}: {status}")

all_ok = all(c[2] for c in checks)
print()
if all_ok:
    print("All checks passed! Run: python src/app.py")
else:
    print("Some checks failed. Fix the issues above first.")
    print("Run: pip install -r requirements.txt")
    if not model_exists:
        print("Run: python download_model.py")

sys.exit(0 if all_ok else 1)