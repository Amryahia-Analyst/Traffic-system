#1st : Red Light Violation Detection - FIXED v2
"""
Red Light Violation Detection - FIXED v2
"""
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2, numpy as np, csv, os
from datetime import datetime
# ══════════════════════════════════════
VIDEO_PATH     = r"../Video_inpt/Red_Light_Violation_1.mp4"
OUTPUT_PATH    = r"../video_out/video_out_red_light_violation.avi"
VIOLATIONS_CSV = r"../VIOLATIONS_CSV/violation.csv"
VIOLATIONS_DIR = r"../VIOLATIONS_Frame/violation.jpg"
STOP_LINE      = [(420, 420), (940, 464)] 
# ══════════════════════════════════════

model   = YOLO(r"../Models/yolo11x_red_light.pt")
tracker = DeepSort(max_age=40, n_init=2)

os.makedirs(VIOLATIONS_DIR, exist_ok=True)
csv_file   = open(VIOLATIONS_CSV, "w", newline="", encoding="utf-8-sig")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["car_id", "frame", "timestamp", "image_path"])


def is_red_light(frame, boxes):
    """كشف الإشارة الحمراء بالـ HSV"""
    if not boxes:
        return False
    for (x1, y1, x2, y2) in boxes:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,120,70]),  np.array([10,255,255])) + \
               cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
        if cv2.countNonZero(mask) / (roi.shape[0]*roi.shape[1]+1e-5) > 0.05:
            return True
    return False


def get_side(px, py):
    """إيجابي = فوق الخط | سالب = تحت الخط"""
    (x1,y1),(x2,y2) = STOP_LINE
    return (x2-x1)*(py-y1) - (y2-y1)*(px-x1)


def draw_ui(frame, red, in_c, out_c, viol_c, fnum):
    h, w = frame.shape[:2]
    # cv2.line(frame, STOP_LINE[0], STOP_LINE[1], (255,0,255), 3)
    # cv2.putText(frame, "STOP LINE", (STOP_LINE[0][0], STOP_LINE[0][1]-8),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    # إشارة
    col = (0,0,255) if red else (0,200,0)
    cv2.rectangle(frame,(10,10),(250,48),col,-1)
    cv2.putText(frame,"Red Light: ON" if red else "Red Light: OFF",
                (15,36),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    # IN/OUT
    cv2.rectangle(frame,(10,58),(240,96),(30,30,30),-1)
    cv2.putText(frame,f"IN: {in_c} | OUT: {out_c}",(15,84),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    # مخالفات
    cv2.rectangle(frame,(10,106),(260,144),(0,0,180),-1)
    cv2.putText(frame,f"Violations: {viol_c}",(15,132),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    # فريم
    cv2.rectangle(frame,(w-100,10),(w-5,42),(0,0,0),-1)
    cv2.putText(frame,str(fnum),(w-95,34),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.putText(frame,datetime.now().strftime("%m/%d/%Y  %H:%M:%S"),
                (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
    return frame


# ══════════════════════════════════════
cap = cv2.VideoCapture(VIDEO_PATH)
W   = int(cap.get(3)); H = int(cap.get(4))
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
OUTPUT_PATH = r"C:\Users\Admin\Desktop\lectures\Capston_Project\Final_project\video_out\first_out.mp4"
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"XVID"), FPS, (W,H))

# ─── STATE ────────────────────────────────────────────────
prev_sides   = {}   # {tid: float}   ← الجانب في الفريم السابق
violated_ids = set()
in_count = out_count = frame_num = 0
# ──────────────────────────────────────────────────────────

print("🚀 Starting...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_num += 1

    # 1. إشارات المرور
    sig_res  = model(frame, classes=[9], verbose=False, conf=0.25)
    sig_boxes = [(int(b.xyxy[0][0]),int(b.xyxy[0][1]),
                  int(b.xyxy[0][2]),int(b.xyxy[0][3]))
                 for r in sig_res for b in r.boxes]
    for (x1,y1,x2,y2) in sig_boxes:
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,165,255),2)

    red_light = is_red_light(frame, sig_boxes)

    # 2. كشف السيارات
    car_res    = model(frame, classes=[2,3,5,7], verbose=False, conf=0.3)
    detections = [([int(b.xyxy[0][0]),int(b.xyxy[0][1]),
                    int(b.xyxy[0][2])-int(b.xyxy[0][0]),
                    int(b.xyxy[0][3])-int(b.xyxy[0][1])],
                   float(b.conf[0]), "car")
                  for r in car_res for b in r.boxes]

    # 3. تتبع
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed(): continue

        tid            = track.track_id
        x1,y1,x2,y2   = map(int, track.to_ltrb())
        cx             = (x1+x2)//2
        cy             = y2          # أسفل السيارة

        curr_side = get_side(cx, cy)

        # ─── FIX: احفظ الجانب السابق قبل أي تعديل ───────
        old_side = prev_sides.get(tid, None)

        # عداد IN/OUT
        if old_side is not None:
            if old_side > 0 and curr_side <= 0:   # عدى من فوق لتحت
                out_count += 1
            elif old_side < 0 and curr_side >= 0: # عدى من تحت لفوق
                in_count  += 1

        # ─── كشف المخالفة ─────────────────────────────────
        # شرط المخالفة: إشارة حمراء + السيارة عدت الخط (من فوق لتحت)
        violated = tid in violated_ids

        if red_light and not violated and old_side is not None:
            if old_side > 0 and curr_side <= 0:
                violated_ids.add(tid)
                violated  = True

                # ─── رسم البوكس الأحمر فوراً قبل حفظ الفريم ───
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.rectangle(frame,(x1,y1-26),(x2,y1),(0,0,255),-1)
                cv2.putText(frame,f"car ID:{tid} Violated",(x1+4,y1-7),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

                now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                img_path  = os.path.join(VIOLATIONS_DIR, f"car{tid}_frame{frame_num}.jpg")
                cv2.imwrite(img_path, frame)
                csv_writer.writerow([tid, frame_num, now, img_path])
                csv_file.flush()
                print(f"🚨 Violation! Car ID:{tid} | Frame:{frame_num}")

        # تحديث الجانب بعد كل الحسابات
        prev_sides[tid] = curr_side

        # رسم (يعاد رسم نفس البوكس الأحمر للسيارات المخالفة فعلاً)
        color = (0,0,255) if violated else (0,255,0)
        label = f"car ID:{tid} Violated" if violated else f"car ID:{tid}"
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.rectangle(frame,(x1,y1-26),(x2,y1),color,-1)
        cv2.putText(frame,label,(x1+4,y1-7),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

    frame = draw_ui(frame, red_light, in_count, out_count, len(violated_ids), frame_num)
    out.write(frame)

    if frame_num % 30 == 0:
        print(f"  Frame {frame_num:5d} | Red:{red_light} | Violations:{len(violated_ids)} | IN:{in_count} OUT:{out_count}")

cap.release(); out.release(); csv_file.close()
print(f"\n✅ Done! Violations: {len(violated_ids)} | Video: {OUTPUT_PATH}")














#2nd : helmet_violation

















import cv2
import cv2
import cvzone
from ultralytics import YOLO
import os
from collections import defaultdict

# =========================
# Load Models
# =========================

# Stage 1: Detect persons and motorcycles
detector = YOLO("../Models/yolo11m.pt")

# Stage 2: Helmet classifier
helmet_model = YOLO("../Models/helmet_violation.pt")

STAGE1_CLASSES = {0: "person", 3: "motorcycle"}
HELMET_CLASSES = {0: "With Helmet", 1: "Without Helmet"}

CONF_STAGE1 = 0.4
CONF_STAGE2 = 0.35

PADDING = 0.25

VIOLATION_COLOR = (0, 0, 255)
SAFE_COLOR = (0, 255, 0)

os.makedirs("Violations", exist_ok=True)
saved_ids = set()
helmet_memory = defaultdict(list)
saved_violations = []

# =========================
# Video Input
# =========================

video_path = "../Video_inpt/Helmet.mp4"

# Use FFmpeg backend
cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error opening video")
    exit()

# =========================
# Video Output
# =========================

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 20

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "output3060(4).mp4",
    fourcc,
    fps,
    (frame_width, frame_height)
)

# =========================
# Helper Functions
# =========================

def expand_box(x1, y1, x2, y2, pad, img_h, img_w):
    w, h = x2 - x1, y2 - y1

    x1 = max(0, int(x1 - w * pad))
    y1 = max(0, int(y1 - h * pad))

    x2 = min(img_w, int(x2 + w * pad))
    y2 = min(img_h, int(y2 + h * pad))

    return x1, y1, x2, y2


def is_likely_rider(person_box, moto_boxes, iou_threshold=0.1):

    px1, py1, px2, py2 = person_box

    for mx1, my1, mx2, my2 in moto_boxes:

        inter_x1 = max(px1, mx1)
        inter_y1 = max(py1, my1)

        inter_x2 = min(px2, mx2)
        inter_y2 = min(py2, my2)

        if inter_x2 > inter_x1 and inter_y2 > inter_y1:

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

            person_area = (px2 - px1) * (py2 - py1)

            if inter_area / person_area > iou_threshold:
                return True

    return False


# =========================
# Main Loop
# =========================

while True:

    success, frame = cap.read()

    if not success or frame is None:
        break

    h, w = frame.shape[:2]

    # =========================
    # Stage 1 Detection
    # =========================

    stage1_results = detector.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0, 3],
        conf=CONF_STAGE1
    )

    person_boxes = []
    moto_boxes = []

    for r in stage1_results:

        for box in r.boxes:

            cls = int(box.cls[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            track_id = -1
            if box.id is not None:
                track_id = int(box.id[0])

            if cls == 0:
                person_boxes.append((x1, y1, x2, y2, track_id))

            elif cls == 3:
                moto_boxes.append((x1, y1, x2, y2))

    # Draw motorcycle boxes
    for mx1, my1, mx2, my2 in moto_boxes:

        cv2.rectangle(
            frame,
            (mx1, my1),
            (mx2, my2),
            (200, 200, 0),
            1
        )

    # =========================
    # Stage 2 Helmet Detection
    # =========================

    for (px1, py1, px2, py2, track_id) in person_boxes:

        # Skip pedestrians
        if moto_boxes and not is_likely_rider(
            (px1, py1, px2, py2),
            moto_boxes
        ):
            continue

        # Expand ROI
        cx1, cy1, cx2, cy2 = expand_box(
            px1, py1, px2, py2,
            PADDING,
            h, w
        )

        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            continue

        helmet_results = helmet_model(
            crop,
            conf=CONF_STAGE2
        )

        for hr in helmet_results:

            for hbox in hr.boxes:

                hcls = int(hbox.cls[0])
                hconf = float(hbox.conf[0])

                helmet_memory[track_id].append(hcls)
                if len(helmet_memory[track_id]) > 10:
                    helmet_memory[track_id].pop(0)

                final_cls = max(set(helmet_memory[track_id]), key=helmet_memory[track_id].count)

                label = HELMET_CLASSES.get(final_cls, "Unknown")
                color = SAFE_COLOR if final_cls == 0 else VIOLATION_COLOR

                # Helmet box
                hx1, hy1, hx2, hy2 = map(int, hbox.xyxy[0])

                # Convert to original frame coordinates
                hx1 += cx1
                hy1 += cy1
                hx2 += cx1
                hy2 += cy1

                display_pad = 20

                bx1 = max(0, hx1 - display_pad)
                by1 = max(0, hy1 - display_pad)
                bx2 = min(w, hx2 + display_pad)
                by2 = min(h, hy2 + display_pad)

                box_w = bx2 - bx1
                box_h = by2 - by1

                cvzone.cornerRect(
                    frame,
                    (bx1, by1, box_w, box_h),
                    l=25,
                    rt=4,
                    colorR=color
                )

                cvzone.putTextRect(
                    frame,
                    f"ID:{track_id} | {label} | {hconf:.2f}",
                    (max(0, bx1), max(50, by1)),
                    scale=2.0,
                    thickness=3,
                    colorR=color,
                    offset=12
                )

                if final_cls == 1:

                    nearest_moto = None
                    best_overlap = 0

                    for mx1, my1, mx2, my2 in moto_boxes:

                        ix1=max(px1,mx1)
                        iy1=max(py1,my1)
                        ix2=min(px2,mx2)
                        iy2=min(py2,my2)

                        if ix2>ix1 and iy2>iy1:
                            overlap=(ix2-ix1)*(iy2-iy1)
                            if overlap>best_overlap:
                                best_overlap=overlap
                                nearest_moto=(mx1,my1,mx2,my2)

                    if nearest_moto is not None:

                        mx1,my1,mx2,my2=nearest_moto

                        already_saved=False

                        for ox1,oy1,ox2,oy2 in saved_violations:

                            ix1=max(mx1,ox1)
                            iy1=max(my1,oy1)
                            ix2=min(mx2,ox2)
                            iy2=min(my2,oy2)

                            if ix2>ix1 and iy2>iy1:

                                inter=(ix2-ix1)*(iy2-iy1)
                                old_area=max(1,(ox2-ox1)*(oy2-oy1))

                                if inter/old_area > 0.5:
                                    already_saved=True
                                    break

                        if not already_saved:

                            saved_violations.append((mx1,my1,mx2,my2))

                            cv2.imwrite(
                                 f"Violations/violation_{track_id}.jpg",
                                    frame

                            )

                            # moto_crop=frame[my1:my2,mx1:mx2]

                            # if moto_crop.size > 0:
                            #     cv2.imwrite(
                            #         f"Violations/moto_{track_id}.jpg",
                            #         moto_crop
                            #     )

    # =========================
    # Save Frame
    # =========================

    out.write(frame)

    # =========================
    # Show Frame
    # =========================

    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# Release Resources
# =========================

cap.release()
out.release()

cv2.destroyAllWindows()

print("Video saved as output.mp4")














# 3rd Speed Violation 
















# =========================================================
# INSTALL REQUIREMENTS
# =========================================================
# افتح الترمينال في VSCode واكتب:
#
# pip install ultralytics supervision opencv-python numpy
#
# =========================================================
# IMPORTS
# =========================================================

import os
import cv2
import numpy as np

from collections import defaultdict, deque
from ultralytics import YOLO

# =========================================================
# PATHS
# =========================================================

VIDEO_PATH = r"../Video_inpt/Video_input_speed_violation_1.mp4"

OUTPUT_PATH = r"../video_out/output_speed_violation3copy.mp4"

VIOLATION_DIR = r"../VIOLATIONS_Frame/speed_violaiton_fram.jpg"

os.makedirs(VIOLATION_DIR, exist_ok=True)

# =========================================================
# LOAD MODELS
# =========================================================

# Vehicle Detection

vehicle_model = YOLO("../Models/yolo11m.pt")

# =========================================================
# VIDEO INFO
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("WIDTH:", width)
print("HEIGHT:", height)

# =========================================================
# VIDEO WRITER
# =========================================================

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

# =========================================================
# SETTINGS
# =========================================================

# المسافة الحقيقية بين الخطين بالمتر
REAL_DISTANCE_METERS = 8

# السرعة القصوى
SPEED_LIMIT = 90

# # خطوط السرعة في Bird Eye
# LINE_UP = 390
# LINE_DOWN = 470
# خطوط السرعة في Bird Eye
LINE_UP = 750
LINE_DOWN = 850

# =========================================================
# BIRD EYE VIEW POINTS
# =========================================================

# SRC = np.float32([

#     [350, 260],   # top-left
#     [1500, 260],  # top-right
#     [1770, 1070], # bottom-right
#     [-390, 1070]  # bottom-left

# ])
SRC = np.float32([

    [500, 310],   # top-left
    [760, 300],  # top-right
    [2000, 1100], # bottom-right
    [-800, 1070]  # bottom-left

])

W = 1000
H = 1200

DST = np.float32([

    [0, 0],
    [W, 0],
    [W, H],
    [0, H]

])

# =========================================================
# PERSPECTIVE MATRIX
# =========================================================

matrix = cv2.getPerspectiveTransform(SRC, DST)

# =========================================================
# TRACKING VARIABLES
# =========================================================

track_history = defaultdict(
    lambda: deque(maxlen=30)
)

# تخزين frame ids
track_frame = {}

# السرعات النهائية
track_speed = {}

# العربيات المخالفة
violated_ids = set()

# =========================================================
# VEHICLE CLASSES
# =========================================================

# car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

# =========================================================
# ROAD MASK FUNCTION
# =========================================================

# def get_road_mask(frame):

#     results = road_model.predict(
#         frame,
#         conf=0.3,
#         verbose=False
#     )

#     mask = np.zeros(
#         (frame.shape[0], frame.shape[1]),
#         dtype=np.uint8
#     )

#     if results[0].masks is not None:

#         masks = results[0].masks.data.cpu().numpy()

#         for m in masks:

#             m = cv2.resize(
#                 m,
#                 (frame.shape[1], frame.shape[0])
#             )

#             mask[m > 0.5] = 255

#     return mask

# =========================================================
# CHECK INSIDE ROAD
# =========================================================

# def inside_road(cx, cy, road_mask):

#     if cx < 0 or cy < 0:
#         return False

#     if cy >= road_mask.shape[0]:
#         return False

#     if cx >= road_mask.shape[1]:
#         return False

#     return road_mask[cy, cx] > 0

# =========================================================
# TRANSFORM POINT
# =========================================================

def transform_point(x, y, matrix):

    point = np.array(
        [[[x, y]]],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        point,
        matrix
    )

    tx = int(transformed[0][0][0])

    ty = int(transformed[0][0][1])

    return tx, ty

# =========================================================
# MAIN LOOP
# =========================================================

frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    overlay = frame.copy()

    # =====================================================
    # DRAW SOURCE POLYGON
    # =====================================================

    # src_pts = SRC.astype(int)

    # cv2.polylines(
    #     overlay,
    #     [src_pts],
    #     True,
    #     (0,255,255),
    #     4
    # )

    # =====================================================
    # ROAD SEGMENTATION
    # =====================================================

    # road_mask = get_road_mask(frame)
    # # Save mask once
    # if frame_id == 1:

    #     mask_vis = cv2.cvtColor(
    #         road_mask,
    #         cv2.COLOR_GRAY2BGR
    #     )

    #     cv2.imwrite(
    #         "road_mask_color.jpg",
    #         mask_vis
    #     )

    # =====================================================
    # VISUALIZE ROAD MASK
    # =====================================================

    # green = np.zeros_like(frame)

    # green[:, :] = (0,255,0)

    # segmented = cv2.bitwise_and(
    #     green,
    #     green,
    #     mask=road_mask
    # )

    # overlay = cv2.addWeighted(
    #     overlay,
    #     1.0,
    #     segmented,
    #     0.25,
    #     0
    # )

    # =====================================================
    # BIRD EYE VIEW
    # =====================================================

    bird_eye = cv2.warpPerspective(
        frame,
        matrix,
        (W, H)
    )

    # =====================================================
    # DRAW BIRD EYE LINES
    # =====================================================

    cv2.line(
        bird_eye,
        (0, LINE_UP),
        (W, LINE_UP),
        (255,0,0),
        3
    )

    cv2.line(
        bird_eye,
        (0, LINE_DOWN),
        (W, LINE_DOWN),
        (0,0,255),
        3
    )

    # =====================================================
    # DETECTION + TRACKING
    # =====================================================

    results = vehicle_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        iou=0.5,
        verbose=False
    )

    # =====================================================
    # CHECK IDS
    # =====================================================

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()

        ids = results[0].boxes.id.cpu().numpy().astype(int)

        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        # =================================================
        # LOOP OBJECTS
        # =================================================

        for box, track_id, cls in zip(
            boxes,
            ids,
            classes
        ):

            if cls not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)

            cy = int((y1 + y2) / 2)

            # =================================================
            # INSIDE ROAD
            # =================================================

            # if not inside_road(
            #     cx,
            #     cy,
            #     road_mask
            # ):
            #     continue

            # =================================================
            # TRANSFORM POINT
            # =================================================

            bx, by = transform_point(
                cx,
                cy,
                matrix
            )

            # =================================================
            # FILTER OUTSIDE BIRD VIEW
            # =================================================

            if bx < 0 or by < 0:
                continue

            if bx >= W or by >= H:
                continue

            # =================================================
            # TRACK HISTORY
            # =================================================

            track_history[track_id].append(
                (bx, by)
            )

            # =================================================
            # DIRECTION
            # =================================================

            direction = "UNKNOWN"

            if len(track_history[track_id]) >= 2:

                prev_y = track_history[
                    track_id
                ][-2][1]

                curr_y = track_history[
                    track_id
                ][-1][1]

                if curr_y > prev_y:
                    direction = "DOWN"
                else:
                    direction = "UP"

            # =================================================
            # SPEED CALCULATION USING FPS
            # =================================================

            if direction == "DOWN":

                # أول خط
                if track_id not in track_frame:

                    if abs(by - LINE_UP) < 10:

                        track_frame[
                            track_id
                        ] = frame_id

                # ثاني خط
                else:

                    if abs(by - LINE_DOWN) < 10:

                        frames_elapsed = (
                            frame_id
                            -
                            track_frame[track_id]
                        )

                        time_elapsed = (
                            frames_elapsed
                            / fps
                        )

                        if time_elapsed > 0:

                            speed = (
                                REAL_DISTANCE_METERS
                                /
                                time_elapsed
                            ) * 3.6

                            track_speed[
                                track_id
                            ] = int(speed)

                            del track_frame[
                                track_id
                            ]

            # =================================================
            # UP DIRECTION
            # =================================================

            elif direction == "UP":

                # أول خط
                if track_id not in track_frame:

                    if abs(by - LINE_DOWN) < 10:

                        track_frame[
                            track_id
                        ] = frame_id

                # ثاني خط
                else:

                    if abs(by - LINE_UP) < 10:

                        frames_elapsed = (
                            frame_id
                            -
                            track_frame[track_id]
                        )

                        time_elapsed = (
                            frames_elapsed
                            / fps
                        )

                        if time_elapsed > 0:

                            speed = (
                                REAL_DISTANCE_METERS
                                /
                                time_elapsed
                            ) * 3.6

                            track_speed[
                                track_id
                            ] = int(speed)

                            del track_frame[
                                track_id
                            ]

            # =================================================
            # CURRENT SPEED
            # =================================================

            current_speed = track_speed.get(
                track_id,
                0
            )

            # =================================================
            # VIOLATION
            # =================================================

            violation = (
                current_speed > SPEED_LIMIT
            )

            # =================================================
            # COLORS
            # =================================================

            color = (0,255,0)

            if violation:
                color = (0,0,255)

            # =================================================
            # DRAW BOX
            # =================================================

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # =================================================
            # SPEED LABEL
            # =================================================

            if current_speed == 0:

                label = (
                    f"ID:{track_id} "
                    f"Calculating... "
                    f"{direction}"
                )

            else:

                label = (
                    f"ID:{track_id} "
                    f"{current_speed} km/h "
                    f"{direction}"
                )

            cv2.putText(
                overlay,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # =================================================
            # CENTER POINT
            # =================================================

            cv2.circle(
                overlay,
                (cx, cy),
                5,
                (255,255,0),
                -1
            )

            # =================================================
            # TRACK TRAIL
            # =================================================

            points = track_history[track_id]

            for i in range(1, len(points)):

                p1 = points[i-1]
                p2 = points[i]

                cv2.line(
                    bird_eye,
                    p1,
                    p2,
                    (255,0,255),
                    2
                )

            # =================================================
            # DRAW POINT IN BIRD VIEW
            # =================================================

            cv2.circle(
                bird_eye,
                (bx, by),
                5,
                color,
                -1
            )

            # =================================================
            # SAVE VIOLATION
            # =================================================
            




            if (
                violation
                and
                track_id not in violated_ids
            ):

                violated_ids.add(track_id)

            # Full frame

                violation_frame = overlay.copy()

                


                cv2.putText(
                    violation_frame,
                    f"Speed Limit: {SPEED_LIMIT} km/h",
                    (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    3
                )


                cv2.putText(
                    violation_frame,
                    f"Vehicle ID: {track_id}",
                    (50,130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )





                cv2.putText(
                    violation_frame,
                    f"Vehicle Speed: {current_speed} km/h",
                    (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )

                cv2.putText(
                    violation_frame,
                    f"VIOLATION",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )
                cv2.imwrite(
                os.path.join(
                    VIOLATION_DIR,
                    f"full_{track_id}_{current_speed}.jpg"
                ),
                violation_frame
                )

                # Vehicle crop
                crop = frame[y1:y2, x1:x2]

                



                

                if crop.size > 0:

                    cv2.imwrite(
                        os.path.join(
                            VIOLATION_DIR,
                            f"crop_{track_id}_{current_speed}.jpg"
                        ),
                        crop
                    )

                print(f"Violation Saved: ID {track_id}")












            # if (
            #     violation
            #     and
            #     track_id not in violated_ids
            # ):

            #     violated_ids.add(track_id)

            #     crop = frame[
            #         y1:y2,
            #         x1:x2
            #     ]

            #     if crop.size > 0:

            #         save_path = os.path.join(
            #             VIOLATION_DIR,
            #             f"car_{track_id}_{current_speed}.jpg"
            #         )

            #         cv2.imwrite(
            #             save_path,
            #             crop
            #         )

            #         print(
            #             f"Violation Saved: "
            #             f"{save_path}"
            #         )

    # =====================================================
    # SPEED LIMIT
    # =====================================================

    cv2.putText(
        overlay,
        f"Speed Limit: {SPEED_LIMIT} km/h",
        (50,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0,255,255),
        3
    )

    # =====================================================
    # FRAME NUMBER
    # =====================================================

    cv2.putText(
        overlay,
        f"Frame: {frame_id}",
        (50,140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        3
    )

    # =====================================================
    # SHOW WINDOWS
    # =====================================================

    preview = cv2.resize(
        overlay,
        (1280, 720)
    )

    bird_preview = cv2.resize(
        bird_eye,
        (700, 900)
    )

    cv2.imshow("Speed Violation Detection", preview)

    cv2.imshow("Bird Eye View", bird_preview)

    # =====================================================
    # WRITE VIDEO
    # =====================================================

    out.write(overlay)

    # =====================================================
    # PRESS Q TO EXIT
    # =====================================================

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

# =========================================================
# RELEASE
# =========================================================

cap.release()

out.release()

cv2.destroyAllWindows()

print("===================================")
print("DONE")
print("OUTPUT:", OUTPUT_PATH)
print("VIOLATIONS:", VIOLATION_DIR)
print("===================================")















# 4th :Wrong Way Detection 











# Wrong Way Detection - Updated & Fixed Version
import os
import cv2
import math
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

VIDEO_PATH = r"../Video_inpt/Wrong_Direction.MP4"
MODEL_PATH = r"../Models/yolo11m.pt"

OUTPUT_DIR = r"../video_out"
VIOLATIONS_DIR = os.path.join(OUTPUT_DIR, "violations")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Error opening video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "wrong_way_output.mp4"),
    fourcc,
    fps,
    (width, height)
)

track_history = defaultdict(lambda: deque(maxlen=25))

violation_count = 0
violated_vehicles = []
violation_buffer = {}  

# تعديل عدد الفريمات المنتظرة (15 فريم تضمن استقرار البوكس وثبات اللقطة)
SAVE_DELAY_FRAMES = 15  
crossed_line_ids = set()

MIN_HISTORY = 20
MIN_MOVEMENT = 40
ANGLE_THRESHOLD = 45
DISTANCE_THRESHOLD = 120
SIZE_THRESHOLD = 50

# =========================================================
# VIOLATION LINE
# =========================================================
VIOLATION_LINE_Y = int(height * 0.65)
VEHICLE_CLASSES = [2, 3, 5, 7]

def already_counted(cx, cy, w, h):
    for car in violated_vehicles:
        distance = np.sqrt((cx - car["cx"]) ** 2 + (cy - car["cy"]) ** 2)
        if (
            distance < DISTANCE_THRESHOLD and
            abs(w - car["w"]) < SIZE_THRESHOLD and
            abs(h - car["h"]) < SIZE_THRESHOLD
        ):
            return True
    return False

# =========================================================
# MAIN PROCESSING LOOP
# =========================================================
while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # قاموس لتخزين إحداثيات البوكس الحالية للسيارات في الفريم الحالي لربطها بالـ Buffer
    current_frame_boxes = {}

    if len(results) > 0:
        result = results[0]
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, ids, classes):
                if cls not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # حفظ الإحداثيات اللحظية للسيارة
                current_frame_boxes[track_id] = (x1, y1, x2, y2)
                track_history[track_id].append((cx, cy))

                status = "NORMAL"
                color = (0, 255, 0)
                history = track_history[track_id]

                if len(history) >= MIN_HISTORY:
                    first_x, first_y = history[0]
                    last_x, last_y = history[-1]
                    movement = abs(last_y - first_y)

                    if movement >= MIN_MOVEMENT:
                        dx = last_x - first_x
                        dy = last_y - first_y
                        angle = math.degrees(math.atan2(abs(dx), abs(dy)))

                        wrong_votes = 0
                        normal_votes = 0

                        for i in range(1, len(history)):
                            step_dy = history[i][1] - history[i - 1][1]
                            if step_dy > 0:
                                wrong_votes += 1
                            elif step_dy < 0:
                                normal_votes += 1

                        is_wrong = (
                            wrong_votes > normal_votes * 2 and
                            dy > 0 and
                            angle < ANGLE_THRESHOLD
                        )

                        if is_wrong:
                            status = "WRONG WAY"
                            color = (0, 0, 255)

                            w = x2 - x1
                            h = y2 - y1

                            if len(history) >= 2:
                                previous_y = history[-2][1]
                                current_y = history[-1][1]

                                crossed_line = (
                                    previous_y < VIOLATION_LINE_Y and
                                    current_y >= VIOLATION_LINE_Y
                                )

                                if (
                                    crossed_line and
                                    track_id not in crossed_line_ids and
                                    not already_counted(cx, cy, w, h)
                                ):
                                    # تسجيل المخالفة وبدء العداد المؤجل
                                    violation_buffer[track_id] = {
                                        "violation_number": violation_count + 1,
                                        "frame_count": 0
                                    }

                                    crossed_line_ids.add(track_id)
                                    violated_vehicles.append({
                                        "cx": cx, "cy": cy, "w": w, "h": h
                                    })
                                    violation_count += 1

                                    # قطع كادر السيارة الفوري (إذا كنت تحتاجه بجانب الفريم الكامل)
                                    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                                    crop_path = os.path.join(VIOLATIONS_DIR, f"car_{violation_count}.jpg")
                                    cv2.imwrite(crop_path, crop)
                                    print(f"[VIOLATION] Vehicle {violation_count} detected.")

                # الرسم القياسي على فريم الفيديو المباشر
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id} {status}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                pts = np.array(history, dtype=np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, color, 2)

    # =========================================================
    # الـ BUFFER LOOP الرئيسي (تم نقله هنا خارج الـ Tracking Loop)
    # =========================================================
    to_delete = []
    for vid, data in violation_buffer.items():
        data["frame_count"] += 1

        # إذا مر الوقت المطلوب، يتم التقاط الصورة بالفريم والبوكس الجديدين
        if data["frame_count"] >= SAVE_DELAY_FRAMES:
            vnum = data["violation_number"]
            
            # نأخذ نسخة من الفريم الحالي (الجديد والمستقر)
            frame_v = frame.copy()
            
            # إذا كانت السيارة لا تزال داخل الكادر، نرسم البوكس الحديث عليها بدقة في الصورة المحفوظة
            if vid in current_frame_boxes:
                vx1, vy1, vx2, vy2 = current_frame_boxes[vid]
                cv2.rectangle(frame_v, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
                cv2.putText(frame_v, f"ID:{vid} WRONG WAY", (vx1, vy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # إضافة لوحة البيانات العلوية السوداء (Overlay)
            cv2.rectangle(frame_v, (0, 0), (width, 140), (0, 0, 0), -1)
            cv2.putText(frame_v, f"Vehicle ID: {vid}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame_v, "WRONG DIRECTION", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame_v, f"Violation Number: {vnum}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # حفظ اللقطة الكاملة والمحدثة
            save_path = os.path.join(VIOLATIONS_DIR, f"violation_{vnum}_id_{vid}.jpg")
            cv2.imwrite(save_path, frame_v)
            print(f"[SAVED] Violation {vnum} saved with stable bounding box.")
            
            to_delete.append(vid)

    for vid in to_delete:
        del violation_buffer[vid]

    # كتابة العداد الكلي على الفيديو
    cv2.putText(frame, f"Violations: {violation_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    video_writer.write(frame)
    cv2.imshow("Wrong Way Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopped by user")
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("\nProcessing Finished")
print(f"Violations Detected = {violation_count}")
print(f"Output Video Saved In: {OUTPUT_DIR}")
print(f"Violation Images Saved In: {VIOLATIONS_DIR}")
























#5th Crop plate

















import cv2
from ultralytics import YOLO
import os

MODEL_PATH = r"../Models/yolo_Crop_plate.pt"
IMAGE_PATH = r"../images_input/English_plate_1.png"
OUTPUT_DIR = r"../image_out"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
img = cv2.imread(IMAGE_PATH)

results = model(img)[0]

for i, box in enumerate(results.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls = int(box.cls[0])

    # Crop and save plate
    crop = img[y1:y2, x1:x2]
    out_path = os.path.join(OUTPUT_DIR, f"plate_{i}_{conf:.2f}.png")
    cv2.imwrite(out_path, crop)

    # Draw annotation on original image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"plate {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print(f"Saved {out_path} | box=({x1},{y1},{x2},{y2}) conf={conf:.2f}")

cv2.imwrite("annotated_frame.png", img)
print("Saved annotated_frame.png")







#6th OCR


from fast_plate_ocr import LicensePlateRecognizer
import cv2

img = cv2.imread(r"../image_out/plate_0_0.75.png")

ocr = LicensePlateRecognizer(
    "cct-s-v2-global-model"
)

result = ocr.run(img)

print(result)


