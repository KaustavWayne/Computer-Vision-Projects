import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

# ==========================================================
# CONFIGURATION
# ==========================================================

PLATE_MODEL_PATH  = "best-car.pt"
VIDEO_INPUT_PATH  = "sample.mp4"
VIDEO_OUTPUT_PATH = "output_video_ocr.avi"

OCR_CONFIDENCE_THRESHOLD = 0.35
USE_GPU = True

# ==========================================================
# LOAD MODELS
# ==========================================================

print("🔄 Loading license plate model...")
plate_model = YOLO(PLATE_MODEL_PATH)

print("🔄 Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=USE_GPU)

# ==========================================================
# OPEN VIDEO
# ==========================================================

cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
if not cap.isOpened():
    print(f"❌ Cannot open: {VIDEO_INPUT_PATH}")
    exit()

fps    = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✅ Video: {width}x{height} @ {fps}fps")

# ==========================================================
# VIDEO WRITER
# ==========================================================

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out    = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))

# ==========================================================
# STABLE OCR STORE
# Key   = track_id  (stable across frames via YOLO tracker)
# Value = {"text": str, "score": float, "votes": dict}
#
# "votes" counts how many times each text was read → picks
# the most-frequent reading, not just the last one.
# This eliminates flicker completely.
# ==========================================================

track_store = {}
# track_store[track_id] = {
#     "best_text"  : str,    ← most-voted text so far
#     "best_score" : float,  ← highest score seen for best_text
#     "votes"      : {text: count}
# }

# ==========================================================
# HELPER: preprocess plate ROI for EasyOCR
# ==========================================================

def preprocess_plate(roi: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

# ==========================================================
# HELPER: update vote store for a track_id
# ==========================================================

def update_store(track_id: int, text: str, score: float):
    if track_id not in track_store:
        track_store[track_id] = {
            "best_text"  : text,
            "best_score" : score,
            "votes"      : {}
        }

    votes = track_store[track_id]["votes"]
    votes[text] = votes.get(text, 0) + 1

    # pick the text with most votes; break ties by score
    best_text  = max(votes, key=lambda t: (votes[t], score if t == text else 0))
    best_score = score if best_text == text else track_store[track_id]["best_score"]

    track_store[track_id]["best_text"]  = best_text
    track_store[track_id]["best_score"] = best_score

# ==========================================================
# HELPER: draw white-box + enlarged-plate overlay
# ==========================================================

TEXT_BOX_H = 80
PLATE_H    = 120
OVERLAY_H  = TEXT_BOX_H + PLATE_H   # 200 px total
OVERLAY_W  = 300

def draw_overlay(frame, x1, y1, x2, y2, plate_roi, plate_text):
    # dynamic width — 3x plate width, clamped
    ov_w = max(200, min(400, (x2 - x1) * 3))
    cx   = (x1 + x2) // 2
    ov_x = cx - ov_w // 2
    ov_y = y1 - OVERLAY_H - 10

    ov_x = max(0, min(ov_x, width  - ov_w))
    ov_y = max(0, min(ov_y, height - OVERLAY_H))

    # ── white text box ────────────────────────────────────
    white_box = np.full((TEXT_BOX_H, ov_w, 3), 255, dtype=np.uint8)

    font, fs, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
    (tw, th), _     = cv2.getTextSize(plate_text, font, fs, thick)
    if tw > ov_w - 20:
        fs = fs * (ov_w - 20) / tw
    (tw, th), _ = cv2.getTextSize(plate_text, font, fs, thick)

    cv2.putText(white_box, plate_text,
                ((ov_w - tw) // 2, (TEXT_BOX_H + th) // 2),
                font, fs, (0, 0, 0), thick, cv2.LINE_AA)

    # ── enlarged plate image ──────────────────────────────
    plate_thumb = cv2.resize(plate_roi, (ov_w, PLATE_H))

    # ── paste into frame ──────────────────────────────────
    frame[ov_y            : ov_y + TEXT_BOX_H,  ov_x : ov_x + ov_w] = white_box
    frame[ov_y + TEXT_BOX_H : ov_y + OVERLAY_H, ov_x : ov_x + ov_w] = plate_thumb

# ==========================================================
# MAIN LOOP  —  uses YOLO tracker for stable IDs
# ==========================================================

frame_count = 0
print("✅ Processing started...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    annotated = frame.copy()

    # ── track=True gives each plate a stable track_id ────
    # botsort is accurate; use bytetrack if botsort errors
    results = plate_model.track(
        frame,
        persist=True,       # ← keeps track IDs consistent across frames
        tracker="botsort.yaml",
        verbose=False
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        out.write(annotated)
        print(f"  Frame {frame_count:05d}", end="\r")
        continue

    for box in results.boxes:

        # ── stable track ID ───────────────────────────────
        if box.id is None:
            continue
        track_id = int(box.id.item())

        # ── bounding box ──────────────────────────────────
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        # ── green rectangle around plate ──────────────────
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── crop plate ROI ────────────────────────────────
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # ── run OCR ───────────────────────────────────────
        ocr_out = reader.readtext(preprocess_plate(roi))

        if ocr_out:
            best      = max(ocr_out, key=lambda r: r[2])
            raw, score = best[1], best[2]

            if score >= OCR_CONFIDENCE_THRESHOLD:
                # clean text
                cleaned = "".join(
                    c for c in raw.upper()
                    if c.isalnum() or c == " "
                ).strip()

                if cleaned:
                    update_store(track_id, cleaned, score)

        # ── get best stable text for this track ───────────
        if track_id in track_store:
            plate_text = track_store[track_id]["best_text"]

            # draw overlay
            draw_overlay(annotated, x1, y1, x2, y2, roi, plate_text)

            # yellow label above box
            cv2.putText(
                annotated, plate_text,
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 255, 255), 2, cv2.LINE_AA
            )

    out.write(annotated)
    print(f"  Frame {frame_count:05d}", end="\r")

# ==========================================================
# RELEASE
# ==========================================================

cap.release()
out.release()

print(f"\n\n✅ Done! → {VIDEO_OUTPUT_PATH}")
print(f"   Total frames: {frame_count}")

# ==========================================================
# PRINT FINAL RESULTS SUMMARY
# ==========================================================

print("\n📋 Detected Plates Summary:")
print("-" * 35)
for tid, data in sorted(track_store.items()):
    votes = data["votes"]
    total = sum(votes.values())
    print(f"  Track {tid:>3} | {data['best_text']:<15} | seen {total} frames")