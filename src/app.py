import os
import sys
import json
import time
import argparse
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # project root (app.py lives in src/)
MODELS_DIR = BASE / "models"
DATA_DIR = BASE / "data"

IMAGENET_LABELS = DATA_DIR / "imagenet_labels.json"
LABEL_TO_BIN = DATA_DIR / "label_to_bin.json"
BINS_FILE = DATA_DIR / "bins.json"
FALLBACK_FILE = DATA_DIR / "fallback_items.json"
COUNTER_FILE = DATA_DIR / "counter.json"
MODEL_FILE = MODELS_DIR / "mobilenetv2-12.onnx"

WINDOW = "BinSorter"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLD = 0.55   # below this a verdict is never auto-accepted
INSTANT_CONFIDENCE = 0.85     # at/above this a single frame is enough
STABLE_HITS = 2               # otherwise N consecutive agreeing frames are required
INFERENCE_INTERVAL = 0.15     # seconds between inferences (time-based, not frame-counted)
IDLE_TIMEOUT = 25             # seconds without activity before returning to IDLE
RESULT_HOLD = 4.0             # seconds a verdict stays on screen
LOW_CONF_AFTER = 2.5          # seconds of scanning without a verdict before offering search
SUPPRESS_CLEAR = 3            # non-matching inferences needed before the same item can re-trigger
MOTION_THRESHOLD = 4.0        # mean abs grey-level diff that counts as movement
MOTION_FRAMES = 2             # consecutive moving frames needed to wake from IDLE

FONT_PATH = None
for p in [
    "C:/Windows/Fonts/seguiemj.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # only works at size 109; last resort
]:
    if os.path.exists(p):
        FONT_PATH = p
        break


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def get_today_key():
    return date.today().isoformat()


def _fresh_counter():
    return {"date": get_today_key(), "sorted": 0, "items": {}}


def load_counter():
    """Return today's counter, resetting it on a new day or if the file is corrupt."""
    data = load_json(COUNTER_FILE, None)
    valid = (
        isinstance(data, dict)
        and data.get("date") == get_today_key()
        and isinstance(data.get("sorted"), int)
        and isinstance(data.get("items"), dict)
    )
    if not valid:
        data = _fresh_counter()
        try:
            save_json(COUNTER_FILE, data)
        except OSError:
            pass
    return data


def increment_counter(label, bin_id):
    data = load_counter()
    data["sorted"] += 1
    key = f"{label}→{bin_id}"
    data["items"][key] = data["items"].get(key, 0) + 1
    try:
        save_json(COUNTER_FILE, data)
    except OSError as e:  # a full/locked disk must never take the kiosk down
        print(f"WARNING: could not save counter: {e}")


def get_top_confused(n=3):
    """Most frequently sorted (label→bin) pairs today."""
    data = load_counter()
    return sorted(data["items"].items(), key=lambda x: x[1], reverse=True)[:n]


get_top_items = get_top_confused


def pretty_label(label):
    return str(label).replace("_", " ")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def route(top5, label_to_bin, threshold=CONFIDENCE_THRESHOLD):
    """Turn a top-5 list of (label, prob) into a verdict dict.

    Aggregates probability by bin across top-5 predictions. When multiple top predictions
    agree on the same waste bin (e.g. mobile phones splitting probability across
    remote_control, hand-held_computer, and cellular_telephone), their confidence is
    combined. The primary label and reason come from the highest-ranked class in that bin.
    """
    bin_probs = {}
    bin_best = {}

    for label, prob in top5:
        info = label_to_bin.get(label)
        if info:
            b = info["bin"]
            bin_probs[b] = bin_probs.get(b, 0.0) + prob
            if b not in bin_best:
                bin_best[b] = (label, prob, info["reason"])

    if not bin_probs:
        best_label, best_prob = top5[0]
        return {"label": best_label, "confidence": best_prob, "bin": None,
                "reason": "Not sure — try the search.", "source": "ai", "low_conf": True}

    best_bin = max(bin_probs.items(), key=lambda x: x[1])[0]
    total_conf = bin_probs[best_bin]
    best_label, _, reason = bin_best[best_bin]

    return {
        "label": best_label,
        "confidence": min(float(total_conf), 1.0),
        "bin": best_bin,
        "reason": reason,
        "source": "ai",
        "low_conf": total_conf < threshold,
    }


def apply_overrides(label_to_bin, overrides, valid_bins):
    """Merge bins.json `overrides` into the label map.

    An override is either `"label": "bin_id"` (keeps the existing reason) or
    `"label": {"bin": "bin_id", "reason": "..."}`.
    """
    merged = dict(label_to_bin)
    for label, ov in (overrides or {}).items():
        if isinstance(ov, str):
            ov = {"bin": ov}
        if not isinstance(ov, dict) or ov.get("bin") not in valid_bins:
            print(f"WARNING: ignoring invalid override for '{label}': {ov}")
            continue
        base = merged.get(label, {})
        merged[label] = {"bin": ov["bin"],
                         "reason": ov.get("reason") or base.get("reason") or "Local rule."}
    return merged


class Classifier:
    def __init__(self, model_path, bins_data=None):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        with open(IMAGENET_LABELS, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        with open(LABEL_TO_BIN, "r", encoding="utf-8") as f:
            self.label_to_bin = json.load(f)
        if bins_data:
            valid = {b["id"] for b in bins_data.get("bins", [])}
            self.label_to_bin = apply_overrides(self.label_to_bin, bins_data.get("overrides"), valid)

    def preprocess(self, frame_bgr):
        # Centre-crop to a square first so objects are not squashed by the resize.
        h, w = frame_bgr.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        x = frame_bgr[y0:y0 + side, x0:x0 + side]
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)

    def classify(self, frame_bgr):
        x = self.preprocess(frame_bgr)
        logits = self.session.run([self.output_name], {self.input_name: x})[0][0]
        probs = softmax(logits)
        top5_idx = probs.argsort()[::-1][:5]
        return [(self.labels[i], float(probs[i])) for i in top5_idx]

    def get_verdict(self, frame_bgr):
        return route(self.classify(frame_bgr), self.label_to_bin)


def load_bins():
    with open(BINS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fallback():
    with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def find_bin(bins_data, bin_id):
    return next((b for b in bins_data["bins"] if b["id"] == bin_id), None)


def hex_to_bgr(hexc, default=(0, 200, 255)):
    try:
        hexc = hexc.lstrip("#")
        r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# Rendering. All colours passed to these helpers are BGR, like the rest of OpenCV.
# --------------------------------------------------------------------------- #

_FONT_CACHE = {}


def _font(size):
    """Cache fonts. Re-loading a colour font per call was a major cost."""
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(FONT_PATH, size) if FONT_PATH else ImageFont.load_default(size)
        except Exception:
            try:
                _FONT_CACHE[size] = ImageFont.load_default(size)
            except Exception:
                _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _rgb(color, default=(255, 255, 255)):
    """BGR(A) tuple -> RGB tuple for PIL. Alpha is ignored."""
    try:
        b, g, r = (int(c) for c in tuple(color)[:3])
        return (r, g, b)
    except Exception:
        return default


class Overlay:
    """Draws all text for one frame in a single BGR->PIL->BGR pass.

    Doing one conversion per string costs ~73 ms each on a 720p frame; batching
    every string into one pass costs ~18 ms total. That is the difference between
    a 2 FPS window that looks frozen and a 57 FPS one.
    """

    def __init__(self, frame_bgr):
        self._img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._d = ImageDraw.Draw(self._img)

    def text(self, s, pos, size=28, color=(255, 255, 255), bg=None):
        s = str(s).replace("\ufe0f", "")   # PIL draws the variation selector as a blank gap
        f = _font(size)
        pos = (int(pos[0]), int(pos[1]))
        if bg is not None:
            bb = self._d.textbbox(pos, s, font=f)
            pad = 6
            self._d.rectangle([bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad],
                              fill=_rgb(bg, (0, 0, 0)))
        self._d.text(pos, s, font=f, fill=_rgb(color), embedded_color=True)
        return self

    def width(self, s, size):
        return self._d.textlength(str(s).replace("\ufe0f", ""), font=_font(size))

    def centred(self, s, cx, y, size=28, color=(255, 255, 255)):
        return self.text(s, (int(cx - self.width(s, size) / 2), int(y)), size, color)

    def wrap(self, s, size, max_w, max_lines=2):
        """Greedy word-wrap; the last line is ellipsised if text overflows."""
        lines, cur = [], ""
        for word in str(s).split():
            trial = f"{cur} {word}".strip()
            if not cur or self.width(trial, size) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1]
            while last and self.width(last + "…", size) > max_w:
                last = last[:-1]
            lines[-1] = last.rstrip() + "…"
        return lines

    def button(self, rect, label, size, fill=(60, 60, 60), color=(255, 255, 255)):
        x0, y0, x1, y1 = (int(v) for v in rect)
        self._d.rounded_rectangle([x0, y0, x1, y1], radius=max((y1 - y0) // 3, 2), fill=_rgb(fill))
        f = _font(size)
        bb = self._d.textbbox((0, 0), label, font=f)
        tx = (x0 + x1) / 2 - (bb[2] - bb[0]) / 2 - bb[0]
        ty = (y0 + y1) / 2 - (bb[3] - bb[1]) / 2 - bb[1]
        self._d.text((int(tx), int(ty)), label, font=f, fill=_rgb(color), embedded_color=True)
        return self

    def frame(self):
        return cv2.cvtColor(np.array(self._img), cv2.COLOR_RGB2BGR)


def draw_text_with_emoji(img_bgr, text, pos, font_size=28, color=(255, 255, 255), bg_color=None):
    """Single-string convenience wrapper."""
    return Overlay(img_bgr).text(text, pos, font_size, color, bg_color).frame()


def _s(h):
    """UI scale factor: all geometry is authored for 720p and scaled to the frame."""
    return max(0.4, h / 720.0)


def search_button_rect(w, h):
    s = _s(h)
    return (w - int(230 * s), int(18 * s), w - int(18 * s), int(78 * s))


def _draw_search_button(t, w, h):
    s = _s(h)
    t.button(search_button_rect(w, h), "🔍 Search (S)", max(int(22 * s), 9), fill=(70, 55, 40))


def guide_box(w, h):
    """The square the user is asked to hold items in: (x0, y0, x1, y1)."""
    s = _s(h)
    cx, cy = w // 2, h // 2
    size = int(min(w, h) / 3 * min(1.0, s * 1.6))
    return cx - size, cy - size, cx + size, cy + size


def crop_to_guide(frame):
    """Classify only what is inside the guide box, not the user's face or the room."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = guide_box(w, h)
    return frame[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]


def draw_guide_box(frame, color=(0, 255, 0)):
    """cv2-only (cheap) guide rectangle with corner accents."""
    h, w = frame.shape[:2]
    s = _s(h)
    x0, y0, x1, y1 = guide_box(w, h)
    t1 = max(int(2 * s), 1)
    t2 = max(int(6 * s), 2)
    arm = int(36 * s)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, t1)
    for (x, y, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(frame, (x, y), (x + dx * arm, y), color, t2)
        cv2.line(frame, (x, y), (x, y + dy * arm), color, t2)
    return frame


def draw_guide(frame, title="Hold item here 👇", subtitle=None):
    h, w = frame.shape[:2]
    s = _s(h)
    draw_guide_box(frame)
    x0, y0, x1, y1 = guide_box(w, h)
    t = Overlay(frame)
    t.centred(title, w // 2, max(y0 - int(48 * s), 4), max(int(26 * s), 10), (80, 255, 80))
    if subtitle:
        t.centred(subtitle, w // 2, min(y1 + int(16 * s), h - int(40 * s)),
                  max(int(20 * s), 9), (220, 220, 220))
    _draw_search_button(t, w, h)
    return t.frame()


def draw_verdict_overlay(frame, verdict, bins_data, extra=None):
    h, w = frame.shape[:2]
    s = _s(h)
    panel = int(220 * s)
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - panel), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

    bin_info = find_bin(bins_data, verdict.get("bin"))
    bar_color = hex_to_bgr(bin_info["color"]) if bin_info else (0, 190, 255)  # amber when unknown
    cv2.rectangle(frame, (0, h - panel), (w, h - panel + max(int(6 * s), 2)), bar_color, -1)

    m = int(30 * s)
    conf = max(0.0, min(float(verdict.get("confidence", 0.0)), 1.0))
    bar_top = h - int(198 * s)
    bar_h = max(int(14 * s), 4)
    bar_w = int((w - 2 * m) * conf)
    cv2.rectangle(frame, (m, bar_top), (m + max(bar_w, 8), bar_top + bar_h), bar_color, -1)
    cv2.rectangle(frame, (m, bar_top), (w - m, bar_top + bar_h), (90, 90, 90), 1)

    t = Overlay(frame)
    big, med, small = max(int(34 * s), 12), max(int(20 * s), 9), max(int(19 * s), 9)
    title = f"{bin_info['emoji']}  {bin_info['label']}" if bin_info else "No bin assigned"
    t.text(title, (m, h - int(172 * s)), big, (255, 255, 255))
    for i, line in enumerate(t.wrap(verdict.get("reason") or "Not sure", med, w - 2 * m, 2)):
        t.text(line, (m, h - int((122 - 26 * i) * s)), med, (215, 215, 215))

    if verdict.get("source") == "fallback":
        label = f"You picked: {pretty_label(verdict['label'])}"
    else:
        label = f"AI: {pretty_label(verdict['label'])}  ({conf * 100:.0f}%)"
        if verdict.get("low_conf"):
            label = "LOW CONFIDENCE - " + label
    t.text(label, (m, h - int(58 * s)), small, (90, 190, 255))

    for item in (extra or []):
        item = tuple(item)
        txt, pos = item[0], item[1]
        size = item[2] if len(item) > 2 else 26
        col = item[3] if len(item) > 3 else (255, 255, 255)
        t.text(txt, (int(pos[0] * s), int(pos[1] * s)), max(int(size * s), 9), col, bg=(20, 20, 20))
    _draw_search_button(t, w, h)
    return t.frame()


def draw_idle(frame, bins_data, counter_data, confused=None):
    h, w = frame.shape[:2]
    s = _s(h)
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    # Bin legend chips along the bottom (cv2 first, text in the overlay pass).
    bins = bins_data.get("bins", [])
    chips = []
    if bins:
        gap = int(12 * s)
        chip_w = (w - gap * (len(bins) + 1)) // len(bins)
        chip_h = int(54 * s)
        y0 = h - int(110 * s)
        for i, b in enumerate(bins):
            x0 = gap + i * (chip_w + gap)
            cv2.rectangle(frame, (x0, y0), (x0 + chip_w, y0 + chip_h), hex_to_bgr(b.get("color", "")), -1)
            chips.append((b, x0 + chip_w // 2, y0 + chip_h // 2, chip_w))

    cx = w // 2
    F = lambda v: max(int(v * s), 10)
    t = Overlay(frame)
    t.centred("BinSorter", cx, h // 2 - int(220 * s), F(56), (255, 255, 255))
    t.centred("Hold your trash up to the camera 👀", cx, h // 2 - int(130 * s), F(28), (210, 210, 210))
    t.centred(f"Items sorted today: {counter_data.get('sorted', 0)}", cx, h // 2 - int(70 * s), F(24), (180, 180, 180))

    y = h // 2 - int(20 * s)
    if confused:
        t.centred("Top items today:", cx, y, F(22), (180, 180, 180))
        y += int(34 * s)
        for item, count in confused:
            label, _, bin_id = item.partition("→")
            b = find_bin(bins_data, bin_id)
            dest = b["label"] if b else bin_id
            t.centred(f"{pretty_label(label)} → {dest}  ×{count}", cx, y, F(20), (160, 160, 160))
            y += int(28 * s)

    for b, ccx, ccy, cw in chips:
        size = F(18)
        txt = f"{b.get('emoji', '')} {b.get('label', b['id'])}"
        while size > 9 and t.width(txt, size) > cw - 8:
            size -= 1
        t.centred(txt, ccx, ccy - size * 0.6, size, (20, 20, 20))

    t.centred("Tap / press any key to start  •  ESC to quit", cx, h - int(40 * s), F(18), (130, 130, 130))
    _draw_search_button(t, w, h)
    return t.frame()


def message_screen(lines, color=(200, 200, 200)):
    """Plain cv2 screen (splash / errors): no PIL needed before the model loads."""
    img = np.zeros((720, 1280, 3), np.uint8)
    y = 330
    for i, line in enumerate(lines):
        scale, thick = (1.6, 3) if i == 0 else (0.8, 2)
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(img, line, ((1280 - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255) if i == 0 else color, thick, cv2.LINE_AA)
        y += th + 30
    return img


# --------------------------------------------------------------------------- #
# Manual search (tkinter)
# --------------------------------------------------------------------------- #

def _text_on(hexc):
    """Black or white text, whichever reads better on the given background."""
    b, g, r = hex_to_bgr(hexc, (80, 80, 80))
    return "#111111" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"


class FallbackUI:
    BG = "#1a1a2e"

    def __init__(self, fallback_data, bins_data):
        self.fallback = fallback_data
        self.bins = bins_data
        self.selection = None
        self._root = None
        self._page = None
        self._on_items = False

    def show(self, parent_frame=None):
        """Blocking 2-tap picker. Returns a verdict dict, or None if cancelled."""
        self.selection = None   # never leak a previous pick into a cancelled search
        self._page = None       # widgets from a previous (destroyed) Tk root are dead
        self._on_items = False
        try:
            import tkinter as tk
        except ImportError:
            print("Manual search needs tkinter, which is not installed.")
            return None
        self._tk = tk
        try:
            root = tk.Tk()
        except tk.TclError as e:
            print(f"Manual search unavailable (no display?): {e}")
            return None
        self._root = root
        root.title("Not sure? Tap to search")
        root.configure(bg=self.BG)
        try:
            root.attributes("-fullscreen", True)
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        root.bind("<Escape>", lambda e: self._back())
        root.protocol("WM_DELETE_WINDOW", root.quit)
        self._show_categories()
        root.after(100, root.focus_force)
        root.mainloop()
        try:
            root.destroy()
        except tk.TclError:
            pass
        self._root = None
        return self.selection

    # -- pages -------------------------------------------------------------- #

    def _new_page(self):
        tk = self._tk
        if self._page is not None:
            self._page.destroy()
        self._page = tk.Frame(self._root, bg=self.BG)
        self._page.pack(fill="both", expand=True, padx=40, pady=20)
        return self._page

    def _grid_buttons(self, parent, entries, cols):
        tk = self._tk
        grid = tk.Frame(parent, bg=self.BG)
        grid.pack(fill="both", expand=True, pady=10)
        for c in range(cols):
            grid.grid_columnconfigure(c, weight=1, uniform="col")
        for i, (text, color, cmd) in enumerate(entries):
            tk.Button(grid, text=text, font=("Segoe UI", 20), bg=color, fg=_text_on(color),
                      activebackground=color, activeforeground=_text_on(color),
                      relief="flat", bd=0, height=2, command=cmd
                      ).grid(row=i // cols, column=i % cols, sticky="nsew", padx=10, pady=8)

    def _footer(self, parent, text, cmd):
        tk = self._tk
        tk.Button(parent, text=text, font=("Segoe UI", 18), command=cmd, bg="#495057", fg="white",
                  activebackground="#5c636a", activeforeground="white", relief="flat",
                  width=14, height=1).pack(pady=20)

    def _show_categories(self):
        tk = self._tk
        self._on_items = False
        page = self._new_page()
        tk.Label(page, text="What kind of item is it?", font=("Segoe UI", 28),
                 bg=self.BG, fg="white").pack(pady=(10, 20))
        entries = []
        for cat_id in self.fallback:
            b = find_bin(self.bins, cat_id)
            if b:
                entries.append((f"{b['emoji']}  {b['label']}", b["color"],
                                lambda c=cat_id: self._show_items(c)))
        self._grid_buttons(page, entries, cols=2)
        self._footer(page, "Cancel  (Esc)", self._root.quit)

    def _show_items(self, category):
        tk = self._tk
        self._on_items = True
        b = find_bin(self.bins, category)
        page = self._new_page()
        tk.Label(page, text=f"{b['emoji']}  {b['label']}", font=("Segoe UI", 26),
                 bg=self.BG, fg="white").pack(pady=(10, 10))
        entries = []
        for item_name, info in self.fallback[category].items():
            color = (find_bin(self.bins, info["bin"]) or b)["color"]
            entries.append((item_name.title(), color,
                            lambda i=item_name, inf=info: self._select(i, inf["bin"], inf["reason"])))
        self._grid_buttons(page, entries, cols=3 if len(entries) > 8 else 2)
        self._footer(page, "Back  (Esc)", self._show_categories)

    def _back(self):
        if self._on_items:
            self._show_categories()
        else:
            self._root.quit()

    def _select(self, item, bin_id, reason):
        self.selection = {"label": item, "confidence": 1.0, "bin": bin_id, "reason": reason,
                          "source": "fallback", "low_conf": False}
        self._root.quit()


# --------------------------------------------------------------------------- #
# Kiosk
# --------------------------------------------------------------------------- #

class MotionDetector:
    """Cheap frame-difference detector used to wake the kiosk without a keypress."""

    def __init__(self):
        self.prev = None

    def update(self, frame):
        g = cv2.cvtColor(cv2.resize(frame, (80, 45), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        score = 0.0 if self.prev is None else float(cv2.absdiff(g, self.prev).mean())
        self.prev = g
        return score


IDLE, DETECT, LOW_CONF, RESULT = "IDLE", "DETECT", "LOW_CONF", "RESULT"
SCANNING = (DETECT, LOW_CONF)


class Kiosk:
    def __init__(self, classifier, bins_data, fallback_ui, auto_wake=True):
        self.clf = classifier
        self.bins = bins_data
        self.fallback_ui = fallback_ui
        self.auto_wake = auto_wake
        self.motion = MotionDetector()
        self.motion_streak = 0

        now = time.time()
        self.state = IDLE
        self.state_since = now
        self.last_activity = now
        self.last_inference = 0.0
        self.verdict = None          # what is on screen
        self.pending = None          # verdict to count when RESULT ends
        self.streak_label, self.streak = None, 0
        self.suppress_label, self.suppress_misses = None, 0

        self.counter = load_counter()
        self.top_items = []
        self.counter_read_at = 0.0

    # -- state helpers ------------------------------------------------------ #

    def enter(self, state, now=None):
        now = time.time() if now is None else now
        self.state = state
        self.state_since = now
        self.last_activity = now
        if state in SCANNING or state == IDLE:
            self.streak_label, self.streak = None, 0
        if state == IDLE:
            self.suppress_label, self.suppress_misses = None, 0
            self.verdict = None

    def show_result(self, verdict, frame=None, now=None):
        if isinstance(frame, (int, float)) and now is None:
            now = frame
            frame = None
        self.verdict = verdict
        self.pending = verdict      # replaces an uncounted AI verdict if the user corrected it
        self.frozen_frame = frame.copy() if frame is not None else None
        self.enter(RESULT, now)

    def commit(self):
        """Count the on-screen verdict exactly once."""
        if self.pending and self.pending.get("bin"):
            increment_counter(self.pending["label"], self.pending["bin"])
            self.counter_read_at = 0.0
        self.pending = None

    def finish_result(self, now=None):
        self.commit()
        if self.verdict and self.verdict.get("source") == "ai":
            # Don't immediately re-sort the item that is still in front of the camera.
            self.suppress_label, self.suppress_misses = self.verdict["label"], 0
        self.frozen_frame = None
        self.enter(DETECT, now)

    # -- per-frame ---------------------------------------------------------- #

    def step(self, frame, now):
        moving = self.motion.update(frame) > MOTION_THRESHOLD
        self.motion_streak = self.motion_streak + 1 if moving else 0

        if self.state == IDLE and self.auto_wake and self.motion_streak >= MOTION_FRAMES:
            self.enter(DETECT, now)
        elif self.state in SCANNING:
            if moving:
                self.last_activity = now
            if now - self.last_activity > IDLE_TIMEOUT:
                self.enter(IDLE, now)

        if self.state in SCANNING and now - self.last_inference >= INFERENCE_INTERVAL:
            self.last_inference = now
            self._infer(frame, now)

        if self.state == RESULT and now - self.state_since > RESULT_HOLD:
            self.finish_result(now)

    def _infer(self, frame, now):
        # Dual-crop: check guide box AND full frame so tall bottles or large items aren't missed
        v_crop = self.clf.get_verdict(crop_to_guide(frame))
        v_full = self.clf.get_verdict(frame)
        if v_crop.get("bin") and not v_full.get("bin"):
            v = v_crop
        elif v_full.get("bin") and not v_crop.get("bin"):
            v = v_full
        else:
            v = v_crop if v_crop.get("confidence", 0) >= v_full.get("confidence", 0) else v_full

        if self.suppress_label:
            if v["label"] == self.suppress_label:
                self.suppress_misses = 0
                self.state_since = now      # don't nag with LOW_CONF while they remove it
                return
            self.suppress_misses += 1
            if self.suppress_misses >= SUPPRESS_CLEAR:
                self.suppress_label = None

        self.verdict = v
        if v["bin"] and not v.get("low_conf"):
            bin_match = (v["bin"] == getattr(self, "streak_bin", None))
            self.streak = self.streak + 1 if bin_match else 1
            self.streak_bin = v["bin"]
            self.streak_label = v["label"]
            # Instant lock on >= INSTANT_CONFIDENCE (85%), or 2 consecutive agreeing frames
            if self.streak >= STABLE_HITS or v["confidence"] >= INSTANT_CONFIDENCE:
                self.show_result(v, frame=frame, now=now)
                return
        else:
            self.streak_bin, self.streak_label, self.streak = None, None, 0

        if self.state == DETECT and now - self.state_since >= LOW_CONF_AFTER:
            self.state = LOW_CONF

    def render(self, frame, now):
        if self.state == IDLE:
            if now - self.counter_read_at > 2.0:
                self.counter = load_counter()
                self.top_items = get_top_confused(3)
                self.counter_read_at = now
            return draw_idle(frame, self.bins, self.counter, self.top_items)

        if self.state == DETECT:
            if self.suppress_label:
                return draw_guide(frame, "Got it! ✅", "Remove the item to scan the next one")
            return draw_guide(frame, "Hold item here 👇", "Scanning…")

        if self.state == LOW_CONF:
            draw_guide_box(frame, (0, 190, 255))
            v = self.verdict or {"label": "nothing", "confidence": 0.0, "bin": None,
                                 "reason": "Hold the item closer, in good light.", "low_conf": True}
            return draw_verdict_overlay(
                frame, v, self.bins,
                extra=[("Not sure yet — hold it closer, or press S / tap Search", (30, 30), 24, (0, 200, 255))],
            )

        # In RESULT state: freeze on the captured frame so it stops right there!
        res_frame = self.frozen_frame.copy() if getattr(self, "frozen_frame", None) is not None else frame.copy()
        remaining = max(0, int(RESULT_HOLD - (now - self.state_since) + 0.99))
        h, w = res_frame.shape[:2]
        s = _s(h)

        # Glowing border in the bin's color
        bin_info = find_bin(self.bins, self.verdict.get("bin"))
        bar_color = hex_to_bgr(bin_info["color"]) if bin_info else (80, 255, 80)
        cv2.rectangle(res_frame, (0, 0), (w, h), bar_color, max(int(6 * s), 3))

        # Top "Got it! ✅" banner
        banner_h = int(60 * s)
        ov = res_frame.copy()
        cv2.rectangle(ov, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.85, res_frame, 0.15, 0, res_frame)

        t = Overlay(res_frame)
        t.centred("Got it! ✅", w // 2, int(12 * s), max(int(32 * s), 16), (80, 255, 80))
        res_frame = t.frame()

        extra = [(f"Next scan in {remaining}s  •  tap / any key to continue", (30, 30), 20, (200, 200, 200))]
        return draw_verdict_overlay(res_frame, self.verdict, self.bins, extra=extra)

    # -- input -------------------------------------------------------------- #

    def search(self, frame):
        sel = self.fallback_ui.show(frame)
        now = time.time()           # the dialog may have been open for a while
        if sel:
            self.show_result(sel, now)
        elif self.state != IDLE:
            self.enter(DETECT, now)

    def poke(self, now):
        """Any key / tap that isn't a specific command."""
        self.last_activity = now
        if self.state == IDLE:
            self.enter(DETECT, now)
        elif self.state == RESULT:
            self.finish_result(now)
        elif self.state == LOW_CONF:
            self.enter(DETECT, now)


def parse_cam_source(val):
    try:
        return int(val)
    except ValueError:
        return str(val)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="BinSorter kiosk")
    ap.add_argument("--cam", type=parse_cam_source, default=0,
                    help="camera index (e.g. 0, 1) or stream URL (e.g. http://192.168.1.5:4747/video)")
    ap.add_argument("--no-flip", action="store_true",
                    help="do not mirror the camera feed (ideal for phone cameras pointing at items)")
    ap.add_argument("--windowed", action="store_true", help="1100x700 window instead of fullscreen")
    ap.add_argument("--force-res", action="store_true", help="force 1280x720 capture (slow on some webcams)")
    ap.add_argument("--no-auto-wake", action="store_true", help="require a key/tap to leave the idle screen")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unknown arguments: {' '.join(unknown)}")
    return args


def _focus_window():
    """Bring the OpenCV window back to the front after the tkinter dialog closes."""
    try:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 0)
    except Exception:
        pass


def main(argv=None):
    args = parse_args(argv)
    t_boot = time.time()
    bins_data = load_bins()
    fallback_data = load_fallback()

    if not MODEL_FILE.exists():
        print(f"Model not found at {MODEL_FILE}")
        print("Run: python download_model.py")
        return 1

    classifier = Classifier(MODEL_FILE, bins_data)
    fallback_ui = FallbackUI(fallback_data, bins_data)
    t_model = time.time()

    fullscreen = not args.windowed
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WINDOW, 1100, 700)

    clicks = []
    cv2.setMouseCallback(WINDOW, lambda ev, x, y, *_: clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONUP else None)

    # Show a window before touching the camera. Opening a webcam and negotiating
    # its format can take ~6s on some drivers; with no window up, that reads as
    # a dead program.
    cv2.imshow(WINDOW, message_screen(["BinSorter", "Starting camera..."]))
    cv2.waitKey(50)
    t_window = time.time()
    # Only trust WND_PROP_VISIBLE for close-detection if this backend reports it.
    can_detect_close = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1

    cap = cv2.VideoCapture(args.cam)
    t_cam = time.time()
    if not cap.isOpened():
        print(f"ERROR: Could not open camera: {args.cam}")
        print("For PC webcams / DroidCam Client, try: python src/app.py --cam=1 (or --cam=2)")
        print("For direct DroidCam Wi-Fi stream, try: python src/app.py --cam=http://<Phone-IP>:4747/video")
        cv2.imshow(WINDOW, message_screen(["BinSorter", f"Could not open camera {args.cam}",
                                           "Check index (--cam=1 or 2)",
                                           "or stream URL (--cam=http://<phone-ip>:4747/video)"],
                                          (90, 90, 255)))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        return 1

    # Do NOT force resolution by default: cap.set() on WIDTH/HEIGHT costs seconds
    # on some webcams. Native resolution is fine because the overlay scales.
    if args.force_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    warm = None
    for _ in range(20):
        ok, warm = cap.read()
        if ok and warm is not None:
            break
        warm = None
        time.sleep(0.1)

    print("BinSorter ready")
    print("  model load  : %.2f s" % (t_model - t_boot))
    print("  window shown: %.2f s" % (t_window - t_model))
    print("  camera open : %.2f s%s" % (t_cam - t_window, "  (forced 720p)" if args.force_res else ""))
    if warm is not None:
        print("  camera feed : %dx%d" % (warm.shape[1], warm.shape[0]))
    else:
        print("  camera feed : no frames yet (will keep retrying)")
    print("  inference   : every %d ms, window %s" % (int(INFERENCE_INTERVAL * 1000),
                                                     "fullscreen" if fullscreen else "windowed"))
    print("  ready in    : %.2f s" % (time.time() - t_boot))
    print("  ESC quit | S search | F fullscreen | any key / tap: start or skip")

    kiosk = Kiosk(classifier, bins_data, fallback_ui, auto_wake=not args.no_auto_wake)
    fail_since = None
    last_reconnect = time.time()

    try:
        while True:
            ret, frame = cap.read()
            now = time.time()
            if not ret or frame is None:
                fail_since = fail_since or now
                lost = now - fail_since
                lines = ["BinSorter", "Camera lost - reconnecting..."] if lost < 3 else \
                        ["BinSorter", "Camera not responding",
                         "Press ESC to quit, or wait for auto-reconnect"]
                cv2.imshow(WINDOW, message_screen(lines, (90, 90, 255)))
                key = cv2.waitKey(50) & 0xFF
                if key == 27 or (can_detect_close and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1):
                    break
                if now - last_reconnect > 5:
                    last_reconnect = now
                    cap.release()
                    cap = cv2.VideoCapture(args.cam)
                continue
            fail_since = None

            if not args.no_flip:
                frame = cv2.flip(frame, 1)
            kiosk.step(frame, now)
            cv2.imshow(WINDOW, kiosk.render(frame, now))
            key = cv2.waitKey(1) & 0xFF

            if can_detect_close and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 27:
                break

            # Mouse / touch: the Search button, or anywhere else = "any key".
            while clicks:
                x, y = clicks.pop(0)
                h, w = frame.shape[:2]
                bx0, by0, bx1, by1 = search_button_rect(w, h)
                if bx0 <= x <= bx1 and by0 <= y <= by1:
                    key = ord("s")
                elif key == 255:
                    key = 0   # treated as a generic poke below

            if key in (ord("s"), ord("S")):
                kiosk.search(frame)
                _focus_window()
            elif key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key != 255:
                kiosk.poke(time.time())
    finally:
        kiosk.commit()   # don't lose a verdict that was on screen when we quit
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    except cv2.error as e:
        print("\nOpenCV error: %s" % e)
        print("If this is a display/GUI error, try: python src/app.py --windowed")
        cv2.destroyAllWindows()
        sys.exit(1)