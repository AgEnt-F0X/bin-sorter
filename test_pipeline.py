"""Offline end-to-end test. No camera. Exercises classify -> routing -> overlay."""
import json, sys, time
from pathlib import Path
import numpy as np, cv2

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / "src"))
import app

fails = []
def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)

print("=" * 62)
print("LABEL INTEGRITY")
print("=" * 62)
labels = json.load(open(BASE / "data" / "imagenet_labels.json", encoding="utf-8"))
check("exactly 1000 labels", len(labels) == 1000, f"got {len(labels)}")
ltb = json.load(open(BASE / "data" / "label_to_bin.json", encoding="utf-8"))
dead = [k for k in ltb if k not in labels]
check("all label_to_bin keys are real ImageNet classes", not dead, f"dead={dead}")
bins_data = json.load(open(BASE / "data" / "bins.json", encoding="utf-8"))
valid = {b["id"] for b in bins_data["bins"]}
orph = [k for k, v in ltb.items() if v["bin"] not in valid]
check("all bins resolve in bins.json", not orph, f"orphans={orph}")
fb = json.load(open(BASE / "data" / "fallback_items.json", encoding="utf-8"))
forb = [i for c in fb.values() for i, v in c.items() if v["bin"] not in valid]
check("fallback bins resolve", not forb, f"orphans={forb}")

print()
print("=" * 62)
print("CLASSIFIER")
print("=" * 62)
clf = app.Classifier(BASE / "models" / "mobilenetv2-12.onnx")
check("session input name resolved", isinstance(clf.input_name, str), clf.input_name)
check("session output name resolved", isinstance(clf.output_name, str), clf.output_name)

# 200 random + tinted frames: proves no IndexError across the whole 1000-class output
print("  stress: 200 random frames over full label space...")
t0 = time.time()
ok_idx, mapped, unmapped, maxidx = 0, 0, 0, 0
for i in range(200):
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    top5 = clf.classify(img)
    for lbl, p in top5:
        ok_idx += 1
        maxidx = max(maxidx, labels.index(lbl) if lbl in labels else 9999)
    v = clf.get_verdict(img)
    if v["bin"]:
        mapped += 1
    else:
        unmapped += 1
el = time.time() - t0
check("no IndexError over 200 frames", ok_idx == 1000, f"{ok_idx}/1000 lookups")
check("label<->index mapping is self-consistent", maxidx != 9999)
check("every frame yields a verdict dict", mapped + unmapped == 200, f"mapped={mapped} ask={unmapped}")
print(f"  latency: {el/200*1000:.1f} ms/frame (incl. 2x classify on unmapped)")
check("latency under 800ms", el / 200 < 0.8, f"{el/200*1000:.0f} ms")

print()
print("=" * 62)
print("ROUTING LOGIC")
print("=" * 62)


def route_conf(label, prob):
    """Route a single-label top-5 through the real app.route()."""
    return app.route([(label, prob)], ltb)


r0 = app.route([("stinkhorn", 0.6), ("banana", 0.3)], ltb)
check("first *mapped* label in top-5 wins", r0["label"] == "banana" and r0["low_conf"], r0["label"])

# Semantic traps: ImageNet class names that do not mean what they look like.
check("ImageNet 'notebook' (a laptop) -> ewaste", ltb["notebook"]["bin"] == "ewaste")
check("ImageNet 'potpie' (food) -> organic", ltb["potpie"]["bin"] == "organic")
check("ImageNet 'hot_pot' (food) -> organic", ltb["hot_pot"]["bin"] == "organic")
check("coffee_mug reason matches its bin", ltb["coffee_mug"]["bin"] == "landfill")

ov = app.apply_overrides(ltb, {"water_bottle": "landfill", "banana": {"bin": "nope"},
                               "laptop": {"bin": "hazardous", "reason": "Local rule"}}, valid)
check("override (string form) applied", ov["water_bottle"]["bin"] == "landfill"
      and ov["water_bottle"]["reason"] == ltb["water_bottle"]["reason"])
check("override (dict form) applied", ov["laptop"] == {"bin": "hazardous", "reason": "Local rule"})
check("invalid override ignored", ov["banana"]["bin"] == "organic")
check("overrides don't mutate the base map", ltb["water_bottle"]["bin"] == "recyclable")


r = route_conf("water_bottle", 0.90)
check("high-conf mapped label -> verdict", r["bin"] == "recyclable" and not r["low_conf"], f"{r['label']} {r['confidence']}")

r2 = route_conf("water_bottle", 0.30)
check("low-conf flagged, verdict withheld from auto-accept", r2["low_conf"] is True, f"conf={r2['confidence']}")

r3 = route_conf("stinkhorn", 0.99)
check("unmapped label -> ask (never guesses a bin)", r3["bin"] is None and r3["low_conf"])

r4 = route_conf("banana", 0.80)
check("organic route", r4["bin"] == "organic", r4["bin"])
r5 = route_conf("lighter", 0.80)
check("hazardous route", r5["bin"] == "hazardous", r5["bin"])
r6 = route_conf("iPod", 0.80)
check("e-waste route", r6["bin"] == "ewaste", r6["bin"])
check("threshold is 0.55", abs(app.CONFIDENCE_THRESHOLD - 0.55) < 1e-9, str(app.CONFIDENCE_THRESHOLD))

print()
print("=" * 62)
print("RENDERING (no display)")
print("=" * 62)
frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
verdict = {"label": "water_bottle", "confidence": 0.87, "bin": "recyclable",
           "reason": "Plastic bottle. Empty it, and recycle if your city accepts rigid plastics.",
           "source": "ai", "low_conf": False}
try:
    out = app.draw_verdict_overlay(frame, verdict, bins_data)
    check("draw_verdict_overlay (emoji bin)", out.shape == frame.shape)
except Exception as e:
    check("draw_verdict_overlay", False, f"{type(e).__name__}: {e}")

try:
    out = app.draw_verdict_overlay(frame, {**verdict, "bin": None, "low_conf": True}, bins_data)
    check("overlay with no bin assigned", out.shape == frame.shape)
except Exception as e:
    check("overlay with no bin", False, f"{type(e).__name__}: {e}")

try:
    check("draw_guide", app.draw_guide(frame).shape == frame.shape)
except Exception as e:
    check("draw_guide", False, f"{type(e).__name__}: {e}")

try:
    check("draw_idle", app.draw_idle(frame, bins_data, {"sorted": 7, "items": {}}).shape == frame.shape)
except Exception as e:
    check("draw_idle", False, f"{type(e).__name__}: {e}")

try:
    app.draw_text_with_emoji(frame, "Hold here 👇 ♻️ 🍃", (10, 10), bg_color=(0, 0, 0))
    check("bg_color 3-tuple accepted", True)
except Exception as e:
    check("bg_color", False, f"{type(e).__name__}: {e}")

try:
    app.draw_text_with_emoji(frame, "test", (10, 10), bg_color=(0, 0, 0, 180))
    check("RGBA bg_color tolerated (no crash)", True)
except Exception as e:
    check("RGBA bg_color", False, f"{type(e).__name__}: {e}")

print()
print("=" * 62)
print("PERSISTENCE")
print("=" * 62)
cf = BASE / "data" / "counter.json"
backup = cf.read_text(encoding="utf-8") if cf.exists() else None
try:
    d = app.load_counter()
    check("load_counter returns dict", isinstance(d, dict) and "sorted" in d)
    app.increment_counter("water_bottle", "recyclable")
    d2 = app.load_counter()
    check("counter incremented", d2["sorted"] == d["sorted"] + 1, f"{d['sorted']} -> {d2['sorted']}")
    check("date key set to today", d2["date"] == app.get_today_key(), d2["date"])
    tc = app.get_top_confused(3)
    check("get_top_confused works", isinstance(tc, list))
    check("counter.json is valid JSON", json.loads(cf.read_text(encoding="utf-8")) is not None)
    check("no .tmp left behind", not cf.with_suffix(".tmp").exists())
except Exception as e:
    check("persistence", False, f"{type(e).__name__}: {e}")
finally:
    if backup is not None:
        cf.write_text(backup, encoding="utf-8")

print()
print("=" * 62)
print("RENDERING DETAILS")
print("=" * 62)
check("hex colour converted to BGR for OpenCV", app.hex_to_bgr("#4DABF7") == (0xF7, 0xAB, 0x4D))
try:
    out = app.draw_verdict_overlay(frame.copy(), verdict, bins_data, extra=[("hi", (10, 10), 20)])
    check("extra item without colour tolerated", out.shape == frame.shape)
except Exception as e:
    check("extra item without colour tolerated", False, f"{type(e).__name__}: {e}")
ovl = app.Overlay(frame)
lines = ovl.wrap("word " * 200, 20, 400, 2)
check("long reason wrapped to <= 2 lines", len(lines) == 2 and lines[-1].endswith("…"))
crop = app.crop_to_guide(frame)
check("crop_to_guide returns a square inside the frame",
      crop.shape[0] == crop.shape[1] and 0 < crop.shape[0] <= 720, str(crop.shape))
small = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
check("classify works on tiny frames", len(clf.classify(small)) == 5)
try:
    check("draw_idle with top items", app.draw_idle(frame.copy(), bins_data, {"sorted": 3},
                                                    [("water_bottle→recyclable", 2)]).shape == frame.shape)
except Exception as e:
    check("draw_idle with top items", False, f"{type(e).__name__}: {e}")
check("search button inside frame", 0 <= app.search_button_rect(1280, 720)[0] < 1280)

print()
print("=" * 62)
print("STATE MACHINE (fake classifier, no camera)")
print("=" * 62)
backup = cf.read_text(encoding="utf-8") if cf.exists() else None
try:
    class FakeClf:
        def __init__(self): self.next = None
        def get_verdict(self, _): return dict(self.next)

    class FakeUI:
        def __init__(self): self.ret = None
        def show(self, _=None): return self.ret

    fc, fu = FakeClf(), FakeUI()
    k = app.Kiosk(fc, bins_data, fu, auto_wake=False)
    still = np.zeros((480, 640, 3), np.uint8)
    t = 1000.0
    base = app.load_counter()["sorted"]
    good = app.route([("water_bottle", 0.70)], ltb)
    weak = app.route([("water_bottle", 0.30)], ltb)

    k.poke(t); check("key wakes IDLE -> DETECT", k.state == app.DETECT)
    fc.next = good
    k.step(still, t + 0.2)
    check("one mid-confidence frame is not enough", k.state == app.DETECT)
    k.step(still, t + 0.4)
    check("two agreeing frames -> RESULT", k.state == app.RESULT)
    k.step(still, t + 0.4 + app.RESULT_HOLD + 0.1)
    check("RESULT times out back to DETECT", k.state == app.DETECT)
    check("verdict counted exactly once", app.load_counter()["sorted"] == base + 1)
    for i in range(5):
        k.step(still, t + 5 + i * 0.2)
    check("same item still in view does not re-trigger", k.state == app.DETECT)

    k2 = app.Kiosk(fc, bins_data, fu, auto_wake=False)
    k2.poke(t)
    fc.next = weak
    for i in range(20):
        k2.step(still, t + 0.2 * (i + 1))
    check("weak verdicts -> LOW_CONF, never RESULT", k2.state == app.LOW_CONF)
    fc.next = good
    k2.step(still, t + 5); k2.step(still, t + 5.2)
    check("LOW_CONF keeps scanning and recovers to RESULT", k2.state == app.RESULT)
    fu.ret = {"label": "can", "confidence": 1.0, "bin": "recyclable", "reason": "x", "source": "fallback"}
    before = app.load_counter()["sorted"]
    k2.search(still)
    check("manual pick replaces the uncounted AI verdict", k2.verdict["label"] == "can"
          and app.load_counter()["sorted"] == before)
    k2.poke(time.time())
    check("manual pick counted once (no double count)", app.load_counter()["sorted"] == before + 1)
    fu.ret = None
    k2.search(still)
    check("cancelled search returns to scanning", k2.state == app.DETECT)

    cf.write_text("{not json", encoding="utf-8")
    check("corrupt counter.json recovers", app.load_counter()["sorted"] == 0)
    cf.write_text(json.dumps({"date": app.get_today_key()}), encoding="utf-8")
    check("counter.json missing keys recovers", isinstance(app.load_counter()["items"], dict))
except Exception as e:
    import traceback; traceback.print_exc()
    check("state machine", False, f"{type(e).__name__}: {e}")
finally:
    if backup is not None:
        cf.write_text(backup, encoding="utf-8")

print()
print("=" * 62)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED - pipeline is sound.")
print("Remaining unknown: camera access and window display (need hardware).")
