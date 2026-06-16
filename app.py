"""
Streamlit App: Traffic Violation & ANPR Pipeline
Combines:
 1) Red Light Violation Detection (YOLO + DeepSORT)
 2) License Plate Cropping (YOLO trained model)
 3) Plate OCR (fast_plate_ocr)
"""

import os
import csv
import tempfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ───────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Traffic Violation & ANPR System", layout="wide")
st.title("🚦 Traffic Violation Detection & ANPR System")

# ───────────────────────────────────────────────────────────
# SIDEBAR: PIPELINE SELECTION
# ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

PIPELINE_OPTIONS = {
    "Red Light Violation Detection (Step 1 only)": "red_light",
    "Crop License Plate (Step 2 only)": "crop_plate",
    "Plate OCR (Step 3 only)": "ocr",
    "Full Pipeline (All 3 Steps)": "all",
}

pipeline_choice = st.sidebar.selectbox(
    "Select pipeline to run:",
    list(PIPELINE_OPTIONS.keys()),
)
mode = PIPELINE_OPTIONS[pipeline_choice]

STOP_LINE_DEFAULT = "(420, 420), (940, 464)"
stop_line_str = st.sidebar.text_input(
    "Stop line coordinates (x1,y1),(x2,y2):", STOP_LINE_DEFAULT
)

red_light_model_path = st.sidebar.text_input(
    "Red light/vehicle YOLO model path:", "yolo11x.pt"
)
plate_model_path = st.sidebar.text_input(
    "License plate YOLO model path:", "yolo_car_plate_trained.pt"
)
ocr_model_name = st.sidebar.text_input(
    "OCR model name:", "cct-s-v2-global-model"
)

# ───────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────

def parse_stop_line(s):
    """Parse '(420, 420), (940, 464)' -> [(420,420),(940,464)]"""
    try:
        s = s.replace(" ", "")
        parts = s.split("),(")
        p1 = parts[0].replace("(", "").replace(")", "").split(",")
        p2 = parts[1].replace("(", "").replace(")", "").split(",")
        return [(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))]
    except Exception:
        return [(420, 420), (940, 464)]


def get_side(px, py, stop_line):
    (x1, y1), (x2, y2) = stop_line
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def is_red_light(frame, boxes):
    if not boxes:
        return False
    for (x1, y1, x2, y2) in boxes:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
               cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        if cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1] + 1e-5) > 0.05:
            return True
    return False


def draw_ui(frame, red, in_c, out_c, viol_c, fnum, stop_line):
    h, w = frame.shape[:2]
    cv2.line(frame, stop_line[0], stop_line[1], (255, 0, 255), 3)
    cv2.putText(frame, "STOP LINE", (stop_line[0][0], stop_line[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    col = (0, 0, 255) if red else (0, 200, 0)
    cv2.rectangle(frame, (10, 10), (250, 48), col, -1)
    cv2.putText(frame, "Red Light: ON" if red else "Red Light: OFF",
                (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(frame, (10, 58), (240, 96), (30, 30, 30), -1)
    cv2.putText(frame, f"IN: {in_c} | OUT: {out_c}", (15, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(frame, (10, 106), (260, 144), (0, 0, 180), -1)
    cv2.putText(frame, f"Violations: {viol_c}", (15, 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(frame, (w - 100, 10), (w - 5, 42), (0, 0, 0), -1)
    cv2.putText(frame, str(fnum), (w - 95, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, datetime.now().strftime("%m/%d/%Y  %H:%M:%S"),
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return frame


# ───────────────────────────────────────────────────────────
# STEP 1: RED LIGHT VIOLATION DETECTION
# ───────────────────────────────────────────────────────────
def run_red_light_detection(video_path, model_path, stop_line, progress_bar, status_text):
    from deep_sort_realtime.deepsort_tracker import DeepSort

    model = YOLO(model_path)
    tracker = DeepSort(max_age=40, n_init=2)

    out_dir = tempfile.mkdtemp(prefix="rl_violations_")
    violations_dir = os.path.join(out_dir, "violation_frames")
    os.makedirs(violations_dir, exist_ok=True)

    output_video_path = os.path.join(out_dir, "violation_output.avi")
    violations_csv_path = os.path.join(out_dir, "violations.csv")

    csv_file = open(violations_csv_path, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["car_id", "frame", "timestamp", "image_path"])

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(3))
    H = int(cap.get(4))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"XVID"), FPS, (W, H))

    prev_sides = {}
    violated_ids = set()
    in_count = out_count = frame_num = 0

    violation_crops = []  # list of (car_id, frame_num, image_path, cropped_image)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        sig_res = model(frame, classes=[9], verbose=False, conf=0.25)
        sig_boxes = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                      int(b.xyxy[0][2]), int(b.xyxy[0][3]))
                     for r in sig_res for b in r.boxes]
        for (x1, y1, x2, y2) in sig_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

        red_light = is_red_light(frame, sig_boxes)

        car_res = model(frame, classes=[2, 3, 5, 7], verbose=False, conf=0.3)
        detections = [([int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                        int(b.xyxy[0][2]) - int(b.xyxy[0][0]),
                        int(b.xyxy[0][3]) - int(b.xyxy[0][1])],
                       float(b.conf[0]), "car")
                      for r in car_res for b in r.boxes]

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx = (x1 + x2) // 2
            cy = y2

            curr_side = get_side(cx, cy, stop_line)
            old_side = prev_sides.get(tid, None)

            if old_side is not None:
                if old_side > 0 and curr_side <= 0:
                    out_count += 1
                elif old_side < 0 and curr_side >= 0:
                    in_count += 1

            violated = tid in violated_ids

            if red_light and not violated and old_side is not None:
                if old_side > 0 and curr_side <= 0:
                    violated_ids.add(tid)
                    violated = True
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    img_path = os.path.join(violations_dir, f"car{tid}_frame{frame_num}.jpg")
                    cv2.imwrite(img_path, frame)
                    csv_writer.writerow([tid, frame_num, now, img_path])
                    csv_file.flush()

                    # crop the violating vehicle for downstream plate steps
                    vx1, vy1 = max(0, x1), max(0, y1)
                    vx2, vy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    crop = frame[vy1:vy2, vx1:vx2].copy()
                    violation_crops.append((tid, frame_num, img_path, crop))

            prev_sides[tid] = curr_side

            color = (0, 0, 255) if violated else (0, 255, 0)
            label = f"car ID:{tid} Violated" if violated else f"car ID:{tid}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 26), (x2, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        frame = draw_ui(frame, red_light, in_count, out_count, len(violated_ids), frame_num, stop_line)
        out.write(frame)

        if frame_num % 5 == 0 or frame_num == total_frames:
            progress_bar.progress(min(frame_num / total_frames, 1.0))
            status_text.text(
                f"Frame {frame_num}/{total_frames} | Red:{red_light} | "
                f"Violations:{len(violated_ids)} | IN:{in_count} OUT:{out_count}"
            )

    cap.release()
    out.release()
    csv_file.close()

    return {
        "output_video": output_video_path,
        "violations_csv": violations_csv_path,
        "violations_dir": violations_dir,
        "violation_crops": violation_crops,
        "in_count": in_count,
        "out_count": out_count,
        "violation_count": len(violated_ids),
    }


# ───────────────────────────────────────────────────────────
# STEP 2: CROP LICENSE PLATES FROM AN IMAGE
# ───────────────────────────────────────────────────────────
def run_plate_cropping(image_bgr, model_path):
    model = YOLO(model_path)
    results = model(image_bgr)[0]

    annotated = image_bgr.copy()
    crops = []  # (crop_image, conf, box)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append((crop, conf, (x1, y1, x2, y2)))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"plate {conf:.2f}", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated, crops


# ───────────────────────────────────────────────────────────
# STEP 3: OCR ON A CROPPED PLATE IMAGE
# ───────────────────────────────────────────────────────────
@st.cache_resource
def load_ocr_model(model_name):
    from fast_plate_ocr import LicensePlateRecognizer
    return LicensePlateRecognizer(model_name)


def run_plate_ocr(plate_img_bgr, model_name):
    ocr = load_ocr_model(model_name)
    result = ocr.run(plate_img_bgr)
    return result


# ───────────────────────────────────────────────────────────
# MAIN UI
# ───────────────────────────────────────────────────────────
stop_line = parse_stop_line(stop_line_str)

if mode == "red_light":
    st.header("🚦 Step 1: Red Light Violation Detection")
    video_file = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        if st.button("Run Detection"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_file.read())
                video_path = tmp.name

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            with st.spinner("Processing video..."):
                results = run_red_light_detection(
                    video_path, red_light_model_path, stop_line, progress_bar, status_text
                )

            st.success(
                f"Done! Violations: {results['violation_count']} | "
                f"IN: {results['in_count']} | OUT: {results['out_count']}"
            )

            st.video(results["output_video"])

            with open(results["violations_csv"], "rb") as f:
                st.download_button("Download violations.csv", f, file_name="violations.csv")

            if results["violation_crops"]:
                st.subheader("Violation Frames")
                for tid, fnum, img_path, crop in results["violation_crops"]:
                    st.image(
                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                        caption=f"Car ID {tid} — Frame {fnum}",
                        width=300,
                    )

            os.unlink(video_path)


elif mode == "crop_plate":
    st.header("✂️ Step 2: License Plate Cropping")
    image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if st.button("Run Plate Detection"):
            with st.spinner("Detecting plates..."):
                annotated, crops = run_plate_cropping(img, plate_model_path)

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated Image")

            if crops:
                st.subheader(f"Detected {len(crops)} Plate(s)")
                cols = st.columns(min(len(crops), 4))
                for i, (crop, conf, box) in enumerate(crops):
                    with cols[i % len(cols)]:
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                 caption=f"Plate {i} (conf={conf:.2f})")
                        _, buf = cv2.imencode(".png", crop)
                        st.download_button(
                            f"Download plate_{i}.png", buf.tobytes(),
                            file_name=f"plate_{i}.png", key=f"dl_{i}"
                        )
            else:
                st.warning("No plates detected.")


elif mode == "ocr":
    st.header("🔤 Step 3: License Plate OCR")
    image_file = st.file_uploader("Upload a cropped plate image", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input Plate", width=300)

        if st.button("Run OCR"):
            with st.spinner("Running OCR..."):
                result = run_plate_ocr(img, ocr_model_name)
            st.success("OCR Result:")
            st.write(result)


elif mode == "all":
    st.header("🔁 Full Pipeline: Red Light Detection → Plate Crop → OCR")
    video_file = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        if st.button("Run Full Pipeline"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_file.read())
                video_path = tmp.name

            # ── Step 1 ──
            st.subheader("Step 1: Red Light Violation Detection")
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            with st.spinner("Running red light detection..."):
                rl_results = run_red_light_detection(
                    video_path, red_light_model_path, stop_line, progress_bar, status_text
                )

            st.success(
                f"Violations found: {rl_results['violation_count']} | "
                f"IN: {rl_results['in_count']} | OUT: {rl_results['out_count']}"
            )
            st.video(rl_results["output_video"])

            with open(rl_results["violations_csv"], "rb") as f:
                st.download_button("Download violations.csv", f, file_name="violations.csv")

            if not rl_results["violation_crops"]:
                st.warning("No violations detected — pipeline stops here.")
            else:
                # ── Step 2 ──
                st.subheader("Step 2: License Plate Cropping")
                plate_results = []  # (car_id, frame_num, plate_crop)

                with st.spinner("Detecting plates on violating vehicles..."):
                    for tid, fnum, img_path, vehicle_crop in rl_results["violation_crops"]:
                        annotated, crops = run_plate_cropping(vehicle_crop, plate_model_path)
                        if crops:
                            # take highest-confidence plate
                            best_crop, best_conf, _ = max(crops, key=lambda c: c[1])
                            plate_results.append((tid, fnum, best_crop, best_conf))
                            st.image(
                                cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB),
                                caption=f"Car ID {tid} — Frame {fnum} (conf={best_conf:.2f})",
                                width=250,
                            )
                        else:
                            st.write(f"Car ID {tid} — Frame {fnum}: no plate detected")

                # ── Step 3 ──
                if plate_results:
                    st.subheader("Step 3: License Plate OCR")
                    with st.spinner("Running OCR on detected plates..."):
                        for tid, fnum, plate_crop, conf in plate_results:
                            text = run_plate_ocr(plate_crop, ocr_model_name)
                            st.write(f"**Car ID {tid} (Frame {fnum})** → Plate: `{text}`")
                else:
                    st.warning("No plates detected for any violating vehicle.")

            os.unlink(video_path)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Pipeline: Red Light Violation Detection → License Plate Cropping → Plate OCR"
)