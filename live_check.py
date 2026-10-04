"""Live camera + window check. Writes a screenshot instead of needing a visible display."""
import sys, time
from pathlib import Path
import numpy as np, cv2

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
import app

cam_arg = sys.argv[1] if len(sys.argv) > 1 else "0"
try:
    cam = int(cam_arg)
except ValueError:
    cam = cam_arg
out = BASE / "live_test.jpg"

cap = cv2.VideoCapture(cam)
if not cap.isOpened():
    print(f"FAIL: camera {cam} will not open")
    print("If using DroidCam client, try camera index 1 or 2 (e.g. python live_check.py 1)")
    print("If using Wi-Fi stream, try: python live_check.py http://<Phone-IP>:4747/video")
    sys.exit(1)
print(f"camera {cam} opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")

bins_data = app.load_bins()
clf = app.Classifier(app.MODEL_FILE, bins_data)

print("running 40 frames of the real pipeline...")
lat = []
disp = None
for i in range(40):
    ok, frame = cap.read()
    if not ok:
        print(f"FAIL: frame {i} read returned False")
        sys.exit(1)
    frame = cv2.flip(frame, 1)
    t0 = time.time()
    v = clf.get_verdict(app.crop_to_guide(frame))   # same region the kiosk classifies
    lat.append(time.time() - t0)
    if i in (0, 39):
        app.draw_guide_box(frame)
        disp = app.draw_verdict_overlay(frame, v, bins_data)
    time.sleep(0.02)

cap.release()
print(f"inference: min {min(lat)*1000:.0f} ms  median {sorted(lat)[len(lat)//2]*1000:.0f} ms  max {max(lat)*1000:.0f} ms")
print(f"sample verdict: {v['label']} @ {v['confidence']*100:.1f}% -> bin={v['bin']}")

cv2.imwrite(str(out), disp)
print(f"wrote screenshot: {out}")

print("trying to open a display window...")
try:
    cv2.namedWindow("BinSorter", cv2.WINDOW_NORMAL)
    cv2.imshow("BinSorter", disp)
    cv2.waitKey(300)
    cv2.destroyAllWindows()
    print("window OK - GUI works on this machine")
except cv2.error as e:
    print(f"window FAILED (headless?): {str(e)[:120]}")
    print("use --windowed flag; pipeline itself is fine")
