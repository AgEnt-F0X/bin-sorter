# BinSorter — "Which bin does this go in?"

Offline, AI-assisted waste-segregation kiosk. Hold any piece of trash in front of a webcam; get
an instant answer: **which bin it belongs in, and why**.

## Run it

```bat
pip install -r requirements.txt
python download_model.py
python verify.py
python src/app.py
```

Or double-click `run.bat`.

## Controls

Works with a keyboard, a mouse, or a touchscreen.

| Key / tap | Action |
|---|---|
| `ESC` (or close the window) | Quit |
| `S` or tap **🔍 Search** | Manual search (fallback taxonomy) — available on every screen |
| `F` | Toggle fullscreen |
| any other key / tap anywhere | Start scanning from idle, skip the result countdown, or restart a scan |

The idle screen also wakes up on its own when someone moves in front of the camera.

## Flags

| Flag | Purpose |
|---|---|
| `--cam=N` or `URL` | Camera index (default `0`, DroidCam client often `1` or `2`) or stream URL (e.g. `http://192.168.1.5:4747/video`) |
| `--no-flip` | Do not mirror the video feed (recommended when using a mobile rear camera pointing down at trash) |
| `--windowed` | Run in a 1100x700 window instead of fullscreen |
| `--force-res` | Force 720p (costs ~6 s of startup on some webcams) |
| `--no-auto-wake` | Don't wake from idle on motion; require a key/tap |

## Troubleshooting

**Window opens but looks frozen.** Check you are not running from a terminal
that grabs focus every keystroke. Also confirm `test_pipeline.py` passes — the
renderer is the usual culprit and it reports its own FPS.

**Window opens behind other windows.** Click the taskbar icon, or use
`--windowed` and drag it forward.

**Blank window, terminal says camera not responding.** Another app holds the
webcam. Close it, or try `--cam=1`. The app retries the camera every 5 s.

**Idle screen keeps waking up on its own.** Flickering lights or a busy
background trigger motion wake. Use `--no-auto-wake`.

**Slow start on an unknown machine.** Startup timings are always printed to the terminal.

## Self-tests

```bat
python verify.py          :: does everything exist and import
python test_pipeline.py   :: offline checks: labels, routing, overrides, rendering, state machine, persistence
python live_check.py      :: real camera + real inference, writes live_test.jpg
```

`test_pipeline.py` needs no camera. It stress-tests 200 frames across the full
1000-class output space, which is what catches label-index mismatches, and drives
the kiosk state machine with a fake classifier.

## How it works

```
IDLE ──motion / key / tap──> DETECT ──2 agreeing confident frames──> RESULT ──4s / tap──> DETECT
                               │                                       ▲
                               └──2.5s, nothing confident──> LOW_CONF ─┤ (keeps scanning)
                                                                │      │
                                                     S / Search └──> manual search
DETECT / LOW_CONF ──25s without movement──> IDLE
```

- **DETECT** — mirrored camera feed and a guide box. Only the **inside of the guide box** is
  classified (not the user's face or the room). Inference runs every 150 ms (time-based).
- **Stability** — a verdict is accepted after 2 consecutive agreeing frames ≥ 55%, or one frame
  ≥ 85%. One noisy frame can't flash a wrong answer.
- **RESULT** — bin emoji + label, confidence bar in the bin's colour, wrapped one/two-line
  reason, countdown. Counted once, when the result ends — so a manual correction replaces it
  instead of double-counting.
- **No double sorting** — after a result, the same item is ignored until it leaves the frame
  ("Got it! Remove the item to scan the next one").
- **LOW_CONF** — confidence < 55%: shows the best guess marked as low-confidence and invites
  manual search, while continuing to scan. Never auto-accepts a weak verdict.
- **Unmapped label** — the AI never invents a bin. Unmapped classes show "No bin assigned".

## Performance

Measured on this machine (i5-7200U, no GPU, 640x480 webcam):

| | |
|---|---|
| Time to visible window | **0.98 s** |
| Inference latency | 10–19 ms, median 16 ms (budget 800 ms) |
| Idle screen render | 19 ms → 53 FPS |
| Guide screen render | 8 ms → 122 FPS |
| Verdict screen render | 12 ms → 85 FPS |
| 1000-class label integrity | verified, 0 dead mappings |
| Mapping coverage | ~120 ImageNet classes → 5 bins |

Two performance traps this build avoids, both of which made an earlier
version look frozen rather than crash:

1. **One PIL conversion pass per frame, not per string.** Each
   `cvtColor`→`Image.fromarray`→`np.array`→`cvtColor` round trip over a
   full frame costs ~73 ms. Six text strings meant six passes = 437 ms/frame
   = 2.3 FPS. The `Overlay` class batches every string for a frame into one
   pass: 19 ms, 53 FPS.
2. **Never force camera resolution.** `cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT)`
   costs ~6 s on some webcams, during which no window exists at all. The
   window now opens first and shows a splash, then the camera is opened.
   The overlay scales to whatever resolution arrives.

Use `--force-res` if you specifically want 720p and can tolerate the startup cost.

## Data files

| File | Purpose |
|---|---|
| `imagenet_labels.json` | 1000 ImageNet-1k classes, correct index order |
| `label_to_bin.json` | ~120 verified ImageNet classes → bin + reason |
| `bins.json` | **the only file an organiser edits** — local bin names, colors, overrides |
| `fallback_items.json` | 2-tap manual taxonomy (~30 items) |
| `counter.json` | persistent daily counter (auto-recovers if corrupted) |

Every key in `label_to_bin.json` is a real ImageNet-1k class name — `test_pipeline.py` asserts
this, because a typo'd key silently becomes a dead mapping. Beware of ImageNet names that don't
mean what they look like: `notebook` is a *laptop*, `potpie`, `hot_pot` and `plate` are *food*.

To retarget to a different city's rules, edit `bins.json` and the `overrides` map. No code changes.
Overrides take either a bin id or a bin + reason:

```json
"overrides": {
  "cup": "landfill",
  "carton": {"bin": "landfill", "reason": "Our city does not accept cartons."}
}
```

Invalid overrides (unknown bin id) are ignored with a warning at startup.

## Requirements

Python 3.8+, webcam, ~150 MB disk. Runs CPU-only — no GPU needed.

## Offline

Zero network calls after model download. Camera frames stay in RAM: nothing uploaded,
nothing written to disk. Counter stores only item labels, never images.

## Known limits

ImageNet-1k has no `battery`, `chip_packet` or `medicine_strip` class — that is why the manual
search exists, and it is the intended path for those items. Lighting dominates accuracy: aim a
lamp at the item and avoid backlighting.
