import cv2
import os
import time
from ultralytics import YOLO
import numpy as np

# ============================================================
# HELMET MODEL OPTIONS (read notes below)
# ============================================================
# OPTION A: Default YOLOv8 (current - no helmet model)
#   local_model = YOLO('yolov8n.pt')
#   helmet_model = None
#
# OPTION B: Use a dedicated helmet YOLOv8 model (RECOMMENDED)
#   Download from: https://universe.roboflow.com/
#   Search: "helmet detection" -> Download YOLOv8 format -> put .pt file in project folder
#   Then set: helmet_model = YOLO('your_helmet_model.pt')
#
# OPTION C: Use YOLOv8 trained on Safety Helmet Dataset (best for traffic)
#   Model link: https://universe.roboflow.com/roboflow-universe-projects/hard-hat-universe-0dy7e
#   Classes: 'helmet', 'no_helmet' or 'head', 'hat' depending on dataset
#
# CURRENT SETUP:
local_model = YOLO('yolov8n.pt')
helmet_model = None   # Set to YOLO('your_helmet_model.pt') if you have one

# ============================================================
# DETECTION CLASSES
# ============================================================
# YOLOv8 COCO class IDs:
# 1  = bicycle  (EXCLUDED - slow moving, no helmet check)
# 3  = motorcycle (INCLUDED - main target)
# NOTE: We only track class 3 (motorcycle), bicycle is excluded automatically
TRACK_CLASS = [3]   # Only motorcycles

# ============================================================
# DIVIDER CONFIGURATION
# ============================================================
DIVIDER_POINTS = [
    (1238, 33),  # Point 1
    (1177, 59),  # Point 2
    (1110, 85),  # Point 3
    (1035, 121),  # Point 4
    (954, 159),  # Point 5
    (845, 218),  # Point 6
    (722, 277),  # Point 7
    (616, 338),  # Point 8
    (485, 416),  # Point 9
    (341, 516),  # Point 10
    (226, 601),  # Point 11
    (108, 694),  # Point 12
    (9, 776)  # Point 13
]

# ============================================================
# DIRECTION CONFIGURATION
# ============================================================
# For DIAGONAL roads:
# Watch ONE side and note the direction of travel
# Options: "UP", "DOWN", "LEFT", "RIGHT",
#          "UP-RIGHT", "UP-LEFT", "DOWN-RIGHT", "DOWN-LEFT"
REFERENCE_SIDE = "LEFT"
REFERENCE_DIRECTION = "UP-RIGHT"  # Diagonal road: left side goes up-right

# ============================================================
# SENSITIVITY SETTINGS
# ============================================================
MOVEMENT_THRESHOLD = 12         # Min pixels to count as movement
MIN_TRACK_FRAMES = 12           # Min frames before direction is judged
DIRECTION_HISTORY_SIZE = 25     # Frames kept in history
DIVIDER_BUFFER_ZONE = 50        # Pixels near divider - no violation flagged
MIN_SPEED_THRESHOLD = 8         # Minimum avg pixels/frame to be a motorcycle
MIN_JOURNEY_FRAMES = 15         # Minimum frames before ANY violation flagged

# ============================================================
# HELMET SETTINGS
# ============================================================
HELMET_CONFIDENCE_THRESHOLD = 0.28   # Lower = catches more no-helmet cases
HELMET_REGION_MULTIPLIER = 0.8

# ============================================================
# TRACKING POINT
# ============================================================
BIKE_TRACKING_PERCENTAGE = 0.70  # 70% down the bounding box

# ============================================================
# FOLDERS
# ============================================================
os.makedirs('violations/no_helmet', exist_ok=True)
os.makedirs('violations/wrong_direction', exist_ok=True)
os.makedirs('violations/combined', exist_ok=True)

# ============================================================
# STATE VARIABLES
# ============================================================
track_history = {}          # Position history per track ID
track_directions = {}       # Direction info per track ID
track_frame_count = {}      # Frame count per track ID (for journey tracking)
track_violations = {}       # Accumulated violations per track ID over full journey
violation_ids = {
    'no_helmet': set(),
    'wrong_direction': set(),
    'combined': set()
}
last_saved_time = {}
COOLDOWN = 3

cap = cv2.VideoCapture("input_videos/sample_traffic.mp4")

frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = int(cap.get(cv2.CAP_PROP_FPS))

print("="*70)
print("TRAFFIC VIOLATION DETECTION - COMPLETE FINAL VERSION")
print("="*70)
print(f"Resolution : {frame_width}x{frame_height} @ {fps}fps")
print(f"Reference  : {REFERENCE_SIDE} side goes {REFERENCE_DIRECTION}")
print(f"Buffer Zone: {DIVIDER_BUFFER_ZONE}px")
print(f"Min Journey: {MIN_JOURNEY_FRAMES} frames before flagging")
print(f"Bicycle    : EXCLUDED")
print("="*70)
print("Controls: 'q'=Quit  'd'=Debug  SPACE=Pause  CLICK Pause button")
print("="*70)

debug_mode = False
paused     = False

##################################################################################################################################
# FUNCTIONS - ALL DEFINED BEFORE MAIN LOOP
##################################################################################################################################

def interpolate_curve(points, num_interpolated=200):
    """Create smooth curve from control points."""
    if len(points) < 2:
        return points
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    t     = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, num_interpolated)
    x_new = np.interp(t_new, t, x_coords)
    y_new = np.interp(t_new, t, y_coords)
    return list(zip(x_new.astype(int), y_new.astype(int)))

def get_side_of_curve(point, curve_points):
    """Determine which side of a curve a point is on."""
    px, py   = point
    min_dist = float('inf')
    side     = "LEFT"
    for i in range(len(curve_points) - 1):
        x1, y1 = curve_points[i]
        x2, y2 = curve_points[i + 1]
        cross   = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        dist    = abs(cross) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + 1e-6)
        if dist < min_dist:
            min_dist = dist
            side = "LEFT" if cross > 0 else "RIGHT"
    return side

def get_distance_to_curve(point, curve_points):
    """Calculate minimum distance from point to curve."""
    px, py = point
    return min(np.sqrt((px - cp[0])**2 + (py - cp[1])**2) for cp in curve_points)

def get_opposite_direction(direction):
    """Return opposite of a direction (supports diagonal)."""
    opposites = {
        "UP"        : "DOWN",
        "DOWN"      : "UP",
        "LEFT"      : "RIGHT",
        "RIGHT"     : "LEFT",
        "UP-RIGHT"  : "DOWN-LEFT",
        "UP-LEFT"   : "DOWN-RIGHT",
        "DOWN-RIGHT": "UP-LEFT",
        "DOWN-LEFT" : "UP-RIGHT"
    }
    return opposites.get(direction, "UNKNOWN")

def calculate_movement_direction(history):
    """
    Calculate direction of movement from position history.
    Supports 8 directions including diagonals for diagonal roads.
    """
    if len(history) < MIN_TRACK_FRAMES:
        return "TRACKING", 0.0

    sample_size     = max(3, len(history) // 3)
    start_positions = history[:sample_size]
    end_positions   = history[-sample_size:]

    start_x = sum(p[0] for p in start_positions) / len(start_positions)
    start_y = sum(p[1] for p in start_positions) / len(start_positions)
    end_x   = sum(p[0] for p in end_positions)   / len(end_positions)
    end_y   = sum(p[1] for p in end_positions)   / len(end_positions)

    dx = end_x - start_x
    dy = end_y - start_y

    displacement = np.sqrt(dx**2 + dy**2)

    if displacement < MOVEMENT_THRESHOLD:
        return "STATIONARY", displacement

    # Angle in degrees: right=0, down=90, left=180/-180, up=-90
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # 8-directional classification with 45-degree sectors
    if   -22.5  <= angle <  22.5:  direction = "RIGHT"
    elif  22.5  <= angle <  67.5:  direction = "DOWN-RIGHT"
    elif  67.5  <= angle < 112.5:  direction = "DOWN"
    elif 112.5  <= angle < 157.5:  direction = "DOWN-LEFT"
    elif  157.5 <= angle <= 180 or -180 <= angle < -157.5:
                                   direction = "LEFT"
    elif -157.5 <= angle < -112.5: direction = "UP-LEFT"
    elif -112.5 <= angle <  -67.5: direction = "UP"
    elif  -67.5 <= angle <  -22.5: direction = "UP-RIGHT"
    else:                          direction = "UNKNOWN"

    return direction, displacement

def get_expected_direction(side):
    """Get expected direction for a side of the divider."""
    if side == REFERENCE_SIDE:
        return REFERENCE_DIRECTION
    else:
        return get_opposite_direction(REFERENCE_DIRECTION)

def is_direction_compatible(detected_dir, expected_dir):
    """
    Check if detected direction is compatible with expected direction.
    Allows diagonal variations so diagonal road traffic is handled correctly.

    Example: if expected is UP-RIGHT, also accept UP and RIGHT as correct.
    """
    if detected_dir in ["TRACKING", "STATIONARY", "BUFFER", "UNKNOWN"]:
        return True  # Don't flag uncertain states

    # Each direction is compatible with itself and its two components
    compatibility_map = {
        "UP"        : ["UP", "UP-LEFT", "UP-RIGHT"],
        "DOWN"      : ["DOWN", "DOWN-LEFT", "DOWN-RIGHT"],
        "LEFT"      : ["LEFT", "UP-LEFT", "DOWN-LEFT"],
        "RIGHT"     : ["RIGHT", "UP-RIGHT", "DOWN-RIGHT"],
        "UP-RIGHT"  : ["UP-RIGHT", "UP", "RIGHT"],
        "UP-LEFT"   : ["UP-LEFT",  "UP", "LEFT"],
        "DOWN-RIGHT": ["DOWN-RIGHT", "DOWN", "RIGHT"],
        "DOWN-LEFT" : ["DOWN-LEFT",  "DOWN", "LEFT"]
    }

    compatible = compatibility_map.get(expected_dir, [expected_dir])
    return detected_dir in compatible

def check_direction_violation(track_id, current_pos, side):
    """Check if vehicle is going in wrong direction."""
    if track_id not in track_history:
        track_history[track_id] = []

    track_history[track_id].append(current_pos)

    if len(track_history[track_id]) > DIRECTION_HISTORY_SIZE:
        track_history[track_id] = track_history[track_id][-DIRECTION_HISTORY_SIZE:]

    detected_direction, displacement = calculate_movement_direction(track_history[track_id])
    expected_direction = get_expected_direction(side)

    track_directions[track_id] = {
        'detected': detected_direction,
        'expected': expected_direction,
        'displacement': displacement
    }

    if detected_direction in ["TRACKING", "STATIONARY"]:
        return False, detected_direction, expected_direction, 0.0

    confidence   = min(1.0, displacement / 50.0)
    is_violation = not is_direction_compatible(detected_direction, expected_direction)

    return is_violation, detected_direction, expected_direction, confidence

def is_slow_vehicle(track_id):
    """
    Returns True if vehicle is moving too slowly to be a motorcycle.
    Used to filter bicycles and slow-moving/stopped vehicles.
    """
    if track_id not in track_history:
        return True
    history = track_history[track_id]
    if len(history) < 5:
        return True   # Not enough data yet - don't flag

    # Calculate average speed over recent frames
    recent = history[-10:]
    total_dist = 0
    for i in range(len(recent) - 1):
        total_dist += np.sqrt(
            (recent[i+1][0] - recent[i][0])**2 +
            (recent[i+1][1] - recent[i][1])**2
        )
    avg_speed = total_dist / max(len(recent) - 1, 1)
    return avg_speed < MIN_SPEED_THRESHOLD

def detect_helmet_improved(frame, box):
    """
    Improved helmet detection supporting:
    - Shiny helmets (glossy)
    - Matte/black helmets (texture + shape based)
    - Half-face helmets
    Rejects: cloth wraps, caps, bare heads

    If helmet_model is set (custom YOLOv8), uses that instead.
    Returns: (has_helmet, confidence)
    """
    x1, y1, x2, y2 = box
    bike_h = y2 - y1
    bike_w = x2 - x1

    # Head region (above bike)
    head_y1 = max(0, int(y1 - bike_h * HELMET_REGION_MULTIPLIER))
    head_y2 = int(y1 + bike_h * 0.4)
    head_x1 = max(0, int(x1 - bike_w * 0.3))
    head_x2 = min(frame.shape[1], int(x2 + bike_w * 0.3))

    head_region = frame[head_y1:head_y2, head_x1:head_x2]

    if head_region.size == 0:
        return False, 0.0

    # If dedicated helmet model is available, use it
    if helmet_model is not None:
        results = helmet_model(head_region, verbose=False)[0]
        for det in results.boxes:
            cls_id = int(det.cls[0])
            conf   = float(det.conf[0])
            label  = results.names[cls_id].lower()
            if ("helmet" in label or "hard" in label or "hat" in label) and \
               "no" not in label and conf > 0.4:
                return True, conf
            if ("no_helmet" in label or "nohelmet" in label or "head" in label) and conf > 0.4:
                return False, conf
        return False, 0.3   # Default: no helmet if model doesn't confirm

    # ------------------------------------------------------------------
    # FALLBACK: Computer vision based detection
    # ------------------------------------------------------------------
    hsv  = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)

    # ----- Method 1: Color Detection -----
    masks = []

    # Bright Red
    masks.append(cv2.inRange(hsv, np.array([0,   100, 100]), np.array([10,  255, 255])))
    masks.append(cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255])))

    # Yellow/Orange
    masks.append(cv2.inRange(hsv, np.array([20, 100, 120]), np.array([35, 255, 255])))

    # Glossy White (very bright)
    masks.append(cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 35, 255])))

    # MATTE/GLOSSY BLACK (any darkness - captures matte black helmets)
    # Key fix: wider range than before to catch non-shiny black helmets
    masks.append(cv2.inRange(hsv, np.array([0, 0, 0]),   np.array([180, 255, 80])))

    # Blue
    masks.append(cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255])))

    # Green
    masks.append(cv2.inRange(hsv, np.array([45, 100, 100]), np.array([75, 255, 255])))

    # DARK GREY / MATTE GREY (common for matte helmets)
    masks.append(cv2.inRange(hsv, np.array([0, 0, 40]),  np.array([180, 40, 150])))

    combined_mask = np.zeros_like(masks[0])
    for m in masks:
        combined_mask = cv2.bitwise_or(combined_mask, m)

    # ----- Method 2: Gloss Detection (shiny helmets) -----
    _, bright_spots = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_ratio    = np.sum(bright_spots > 0) / max(gray.size, 1)
    has_gloss       = bright_ratio > 0.02

    # ----- Method 3: Texture Analysis -----
    # Helmets (even matte) are more uniform than cloth/hair
    mean_f    = cv2.blur(gray.astype(float), (5, 5))
    sqr_f     = cv2.blur((gray.astype(float))**2, (5, 5))
    variance  = np.maximum(sqr_f - mean_f**2, 0)
    avg_text  = np.mean(np.sqrt(variance))
    # Matte helmets: texture < 35 (slightly higher than glossy <25)
    is_smooth = avg_text < 35

    # ----- Method 4: Shape (dome/rounded) -----
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges     = cv2.Canny(blurred, 30, 100)  # Lower threshold catches matte edges too
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    has_dome_shape = False
    max_roundness  = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 150:  # Slightly lower min to catch partially visible helmets
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                max_roundness = max(max_roundness, circularity)
                if circularity > 0.45:  # Slightly lower threshold for matte helmets
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cy = int(M["m01"] / M["m00"])
                        if cy < head_region.shape[0] * 0.55:
                            has_dome_shape = True

    # ----- Method 5: Size Check -----
    helmet_pixels      = np.sum(combined_mask > 0)
    total_pixels       = head_region.shape[0] * head_region.shape[1]
    color_ratio        = helmet_pixels / max(total_pixels, 1)
    has_sufficient_size = helmet_pixels > 250

    # ----- Method 6: Darkness check for MATTE BLACK helmets -----
    # Matte black helmets are very dark in ALL channels
    dark_mask   = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 70]))
    dark_ratio  = np.sum(dark_mask > 0) / max(total_pixels, 1)
    is_very_dark = dark_ratio > 0.25   # >25% dark pixels = likely black helmet

    # ----- Scoring -----
    color_score  = min(1.0, color_ratio / 0.15) if has_sufficient_size else 0.0
    gloss_score  = 1.0 if has_gloss else 0.0
    smooth_score = 1.0 if is_smooth else 0.0
    shape_score  = min(1.0, max_roundness / 0.5) if has_dome_shape else 0.0
    dark_score   = 1.0 if is_very_dark else 0.0

    # Weighted combination
    final_confidence = (
        color_score  * 0.25 +
        gloss_score  * 0.20 +
        smooth_score * 0.20 +
        shape_score  * 0.20 +
        dark_score   * 0.15
    )

    # ------------------------------------------------------------------
    # DECISION LOGIC
    # Key insight: cloth wraps FAIL the gloss+smooth test AND the shape test
    # Helmets (any type) PASS at least one of: gloss, smooth surface, dark+shape
    # ------------------------------------------------------------------

    # Path 1: Standard helmet (shiny/coloured/white)
    # Needs: confidence + surface quality + either colour or shape
    standard_helmet = (
        final_confidence > HELMET_CONFIDENCE_THRESHOLD and
        (gloss_score > 0.3 or smooth_score > 0.5) and   # some surface quality
        (color_score > 0.2 or shape_score > 0.3) and    # colour OR shape evidence
        has_sufficient_size
    )

    # Path 2: Matte / black helmet
    # Very dark + reasonably smooth + dome shape visible
    matte_black_helmet = (
        is_very_dark and
        is_smooth and
        has_sufficient_size and
        shape_score > 0.25          # Only need weak shape evidence
    )

    # Path 3: Half-face or partially visible helmet
    # Strong shape evidence alone is enough if there is some colour match
    half_face_helmet = (
        shape_score > 0.55 and      # Clear dome shape
        color_score > 0.15 and      # Some helmet colour
        has_sufficient_size
    )

    # Cloth wrap / cap REJECTION check:
    # Cloth tends to be: NOT glossy, HIGH texture, NO dome shape, NOT very dark
    # If ALL of these are true → it is cloth, not a helmet
    is_cloth = (
        gloss_score < 0.3 and       # No gloss
        smooth_score < 0.3 and      # Textured surface
        shape_score < 0.25 and      # No dome shape
        not is_very_dark            # Not a dark helmet
    )

    has_helmet = (
        not is_cloth and
        (standard_helmet or matte_black_helmet or half_face_helmet)
    )

    return has_helmet, final_confidence

def save_violation_snapshot(frame, track_id, violations, direction_info=None):
    """Save snapshot ONLY for violations."""
    timestamp = int(time.time() * 1000)
    annotated = frame.copy()

    if direction_info:
        text = (f"ID:{track_id} | Side:{direction_info['side']} | "
                f"Going:{direction_info['detected']} | Expected:{direction_info['expected']}")
        cv2.putText(annotated, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for violation_type in violations:
        cooldown_key = f"{violation_type}_{track_id}"

        if cooldown_key not in last_saved_time or \
           (time.time() - last_saved_time[cooldown_key]) > COOLDOWN:

            if len(violations) > 1:
                folder   = 'violations/combined'
                filename = (f"{folder}/ID{track_id}_"
                            f"{'_'.join(violations)}_{timestamp}.jpg")
            else:
                folder   = f'violations/{violation_type}'
                filename = f"{folder}/ID{track_id}_{timestamp}.jpg"

            cv2.imwrite(filename, annotated)
            last_saved_time[cooldown_key] = time.time()
            print(f"📸 Saved: {filename}")

def draw_pause_button(frame, paused):
    """Draw on-screen pause/play button."""
    bx, by, bw, bh = frame_width - 120, 20, 100, 40
    color = (50, 50, 200) if paused else (50, 200, 50)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
    cv2.putText(frame, "PLAY" if paused else "PAUSE",
                (bx + 15, by + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return bx, by, bw, bh

def mouse_callback(event, x, y, flags, param):
    """Handle mouse click on pause button."""
    global paused
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = param
        if bx <= x <= bx + bw and by <= y <= by + bh:
            paused = not paused
            print(f"{'Paused' if paused else 'Playing'}")

##################################################################################################################################
# MAIN LOOP
##################################################################################################################################

smooth_curve  = interpolate_curve(DIVIDER_POINTS, num_interpolated=200)
cv2.namedWindow("Traffic Monitor")
button_coords = (0, 0, 0, 0)
frame_count   = 0

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

    display_frame = frame.copy()

    # Draw curved divider
    for i in range(len(smooth_curve) - 1):
        cv2.line(display_frame, smooth_curve[i], smooth_curve[i + 1], (0, 255, 255), 3)

    if debug_mode:
        for i, pt in enumerate(DIVIDER_POINTS):
            cv2.circle(display_frame, pt, 8, (0, 0, 255), -1)
            cv2.putText(display_frame, f"P{i+1}", (pt[0]+10, pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Track ONLY motorcycles (class 3) - bicycles (class 1) automatically excluded
    results = local_model.track(frame, persist=True, classes=TRACK_CLASS, verbose=False)[0]

    if results.boxes.id is not None:
        boxes     = results.boxes.xyxy.int().cpu().tolist()
        track_ids = results.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Tracking point: 70% down the bounding box
            bike_track_x = (x1 + x2) // 2
            bike_track_y = int(y1 + (y2 - y1) * BIKE_TRACKING_PERCENTAGE)

            # Increment frame count for this track
            track_frame_count[track_id] = track_frame_count.get(track_id, 0) + 1

            # Skip slow-moving vehicles (bicycles, stopped bikes)
            if is_slow_vehicle(track_id):
                if debug_mode:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(display_frame, "SLOW/SKIP", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                continue

            # Determine side using 70% tracking point
            side = get_side_of_curve((bike_track_x, bike_track_y), smooth_curve)

            # Uncomment below if sides appear inverted for your video:
            # if side == "LEFT":  side = "RIGHT"
            # elif side == "RIGHT": side = "LEFT"

            # Distance to divider for buffer zone
            dist_to_divider = get_distance_to_curve((bike_track_x, bike_track_y), smooth_curve)
            is_near_divider = dist_to_divider < DIVIDER_BUFFER_ZONE

            # -------------------------------------------------------
            # DIRECTION CHECK
            # Only check after minimum journey frames observed
            # This prevents early false detections
            # -------------------------------------------------------
            has_enough_journey = track_frame_count[track_id] >= MIN_JOURNEY_FRAMES

            if not is_near_divider and has_enough_journey:
                direction_violation, detected_dir, expected_dir, dir_confidence = \
                    check_direction_violation(track_id, (bike_track_x, bike_track_y), side)
            else:
                direction_violation = False
                detected_dir = "BUFFER" if is_near_divider else "WAITING"
                expected_dir = get_expected_direction(side)
                dir_confidence = 0.0
                # Still update history even if not checking violation
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((bike_track_x, bike_track_y))
                if len(track_history[track_id]) > DIRECTION_HISTORY_SIZE:
                    track_history[track_id] = track_history[track_id][-DIRECTION_HISTORY_SIZE:]

            # -------------------------------------------------------
            # HELMET CHECK
            # -------------------------------------------------------
            has_helmet, helmet_conf = detect_helmet_improved(frame, box)
            is_no_helmet = not has_helmet

            # -------------------------------------------------------
            # ACCUMULATE VIOLATIONS OVER THE JOURNEY
            # Only flag if track has been observed for minimum frames
            # -------------------------------------------------------
            if track_id not in track_violations:
                track_violations[track_id] = {
                    'no_helmet_count'  : 0,
                    'wrong_dir_count'  : 0,
                    'total_frames'     : 0,
                    'side'             : side,
                    'detected_dir'     : detected_dir,
                    'expected_dir'     : expected_dir,
                    'helmet_conf_sum'  : 0.0,
                    'dir_conf_sum'     : 0.0,
                }

            tv = track_violations[track_id]
            tv['total_frames']    += 1
            tv['side']             = side
            tv['detected_dir']     = detected_dir
            tv['expected_dir']     = expected_dir
            tv['helmet_conf_sum'] += helmet_conf
            tv['dir_conf_sum']    += dir_confidence

            if is_no_helmet:
                tv['no_helmet_count'] += 1
            if direction_violation and dir_confidence > 0.55:   # lowered from 0.75
                tv['wrong_dir_count'] += 1

            # -------------------------------------------------------
            # DECIDE VIOLATION based on journey majority
            # -------------------------------------------------------
            violations = []
            total      = max(tv['total_frames'], 1)

            no_helmet_ratio  = tv['no_helmet_count']  / total
            wrong_dir_ratio  = tv['wrong_dir_count']  / total

            if has_enough_journey:
                # No helmet: >35% of frames show no helmet
                # (lower than before so cloth wraps are caught)
                if no_helmet_ratio > 0.35:
                    violations.append("no_helmet")
                    violation_ids['no_helmet'].add(track_id)

                # Wrong direction: >30% of frames show wrong direction
                # (lower than 40% so genuine wrong-way drivers are caught)
                if wrong_dir_ratio > 0.30:
                    violations.append("wrong_direction")
                    violation_ids['wrong_direction'].add(track_id)

                if len(violations) > 1:
                    violation_ids['combined'].add(track_id)

            # -------------------------------------------------------
            # DRAWING
            # -------------------------------------------------------
            bike_h       = y2 - y1
            rider_y_top  = max(0, int(y1 - bike_h * HELMET_REGION_MULTIPLIER))

            if not violations:
                status = "OK" if has_enough_journey else "..."
                color  = (0, 255, 0)
            else:
                parts = []
                if "no_helmet"      in violations: parts.append("No Helmet")
                if "wrong_direction" in violations: parts.append("Wrong Way")
                status = " | ".join(parts)
                color  = (0, 0, 255)

                direction_info = {
                    'side'    : side,
                    'detected': detected_dir,
                    'expected': expected_dir
                }
                save_violation_snapshot(display_frame, track_id, violations, direction_info)

            cv2.rectangle(display_frame, (x1, rider_y_top), (x2, y2), color, 2)
            cv2.putText(display_frame, status, (x1, rider_y_top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if debug_mode:
                cv2.circle(display_frame, (cx, cy), 5, (255, 0, 255), -1)

                t_color = (255, 165, 0) if is_near_divider else (0, 255, 0)
                cv2.circle(display_frame, (bike_track_x, bike_track_y), 8, t_color, -1)

                dbg = (f"ID:{track_id}|{side}|{detected_dir}"
                       f"|H:{helmet_conf:.2f}|D:{int(dist_to_divider)}"
                       f"|F:{track_frame_count.get(track_id,0)}"
                       f"|WD:{tv['wrong_dir_count']}/{total}"
                       f"|NH:{tv['no_helmet_count']}/{total}")
                cv2.putText(display_frame, dbg, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

                if track_id in track_history and len(track_history[track_id]) > 1:
                    pts = track_history[track_id]
                    for i in range(len(pts) - 1):
                        cv2.line(display_frame, pts[i], pts[i+1], (0, 255, 0), 2)

    # Dashboard
    cv2.rectangle(display_frame, (10, 10), (420, 145), (0, 0, 0), -1)
    cv2.rectangle(display_frame, (10, 10), (420, 145), (255, 255, 255), 2)
    yo = 30
    cv2.putText(display_frame, f"Frame: {frame_count}", (20, yo),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    yo += 25
    cv2.putText(display_frame, f"No Helmet      : {len(violation_ids['no_helmet'])}",
                (20, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    yo += 25
    cv2.putText(display_frame, f"Wrong Direction: {len(violation_ids['wrong_direction'])}",
                (20, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    yo += 25
    cv2.putText(display_frame, f"Combined       : {len(violation_ids['combined'])}",
                (20, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    yo += 25
    cv2.putText(display_frame, f"Debug: {'ON' if debug_mode else 'OFF'} (d)",
                (20, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    button_coords = draw_pause_button(display_frame, paused)
    cv2.setMouseCallback("Traffic Monitor", mouse_callback, button_coords)
    cv2.imshow("Traffic Monitor", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if   key == ord('q'):  break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug: {'ON' if debug_mode else 'OFF'}")
    elif key == ord(' '):
        paused = not paused
        print(f"{'Paused' if paused else 'Playing'}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("FINAL STATISTICS")
print("="*70)
print(f"No Helmet Violations   : {len(violation_ids['no_helmet'])}")
print(f"Wrong Direction        : {len(violation_ids['wrong_direction'])}")
print(f"Combined Violations    : {len(violation_ids['combined'])}")
print("="*70)
print("Snapshots saved in 'violations/' - only actual violations saved.")
print()
print("HELMET MODEL RECOMMENDATION:")
print("For better accuracy, use a dedicated YOLOv8 helmet model.")
print("Download from: https://universe.roboflow.com/")
print("Search: 'helmet detection' -> Download YOLOv8 format")
print("Then set: helmet_model = YOLO('your_model.pt') at top of file")
