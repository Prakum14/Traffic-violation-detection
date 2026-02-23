import cv2
import os
import time
from ultralytics import YOLO
import numpy as np

# ============================================================
# HELMET-ONLY DETECTION VERSION
# Wrong-way detection is DISABLED - focus only on helmets
# ============================================================

local_model = YOLO('yolov8n.pt')
helmet_model = None  # Set to YOLO('helmet_model.pt') if you have one

TRACK_CLASS = [3]  # Only motorcycles

# ============================================================
# HELMET TUNING PARAMETERS - BALANCED
# ============================================================
HELMET_CONFIDENCE_THRESHOLD = 0.32   # Middle ground (was 0.35, originally 0.30)
HELMET_REGION_MULTIPLIER = 0.8

# Individual method weights (must sum to 1.0)
WEIGHT_COLOR = 0.25
WEIGHT_GLOSS = 0.20
WEIGHT_SMOOTH = 0.15    # Reduced from 0.20 - less weight on smoothness
WEIGHT_SHAPE = 0.25     # Increased from 0.20 - more weight on dome shape
WEIGHT_DARK = 0.15

# Detection thresholds for each path
# Path 1: Standard helmet (shiny/colored)
PATH1_GLOSS_MIN = 0.2      # Minimum gloss score
PATH1_SMOOTH_MIN = 0.4     # Minimum smooth score
PATH1_COLOR_MIN = 0.15     # Minimum color score
PATH1_SHAPE_MIN = 0.25     # Minimum shape score

# Path 2: Matte/black helmet - BALANCED
PATH2_DARK_THRESHOLD = 0.35     # Very dark required (balanced)
PATH2_SMOOTH_THRESHOLD = 30     # Very smooth required (balanced)
PATH2_SHAPE_MIN = 0.18          # Some dome shape required (lowered from 0.40)

# Path 3: Half-face helmet
PATH3_SHAPE_MIN = 0.5      # Clear dome shape needed
PATH3_COLOR_MIN = 0.12     # Some color needed

# Rejection thresholds - AGGRESSIVE to catch no-helmet cases
CLOTH_GLOSS_MAX = 0.30     # Increased from 0.25
CLOTH_SMOOTH_MAX = 0.40    # Increased from 0.35
CLOTH_SHAPE_MAX = 0.35     # Increased from 0.3
CLOTH_COLOR_MAX = 0.45     # Increased from 0.4

BARE_COLOR_MAX = 0.25      # Increased from 0.2
BARE_SHAPE_MAX = 0.30      # Increased from 0.25

MIN_HELMET_PIXELS = 250

# Journey-based detection - BALANCED
MIN_JOURNEY_FRAMES = 12             # Back to 12 frames
NO_HELMET_RATIO_THRESHOLD = 0.22    # Flag if >22% frames show no helmet (middle ground)

# ============================================================
# SETUP
# ============================================================
os.makedirs('violations/no_helmet', exist_ok=True)
os.makedirs('violations/debug_analysis', exist_ok=True)

track_history = {}
track_frame_count = {}
track_violations = {}
violation_ids = set()
last_saved_time = {}
COOLDOWN = 3

cap = cv2.VideoCapture("input_videos/sample_traffic.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("="*70)
print("HELMET DETECTION - FOCUSED VERSION")
print("="*70)
print(f"Resolution: {frame_width}x{frame_height} @ {fps}fps")
print(f"Threshold: {HELMET_CONFIDENCE_THRESHOLD}")
print(f"Min Journey: {MIN_JOURNEY_FRAMES} frames")
print(f"Violation Ratio: >{NO_HELMET_RATIO_THRESHOLD*100}%")
print("="*70)
print("Controls: 'q'=Quit  'd'=Debug  's'=Screenshot  SPACE=Pause")
print("="*70)

debug_mode = True  # Start with debug ON
paused = False
mouse_x, mouse_y = 0, 0

##################################################################################################################################

def draw_pause_button(frame, paused):
    """Draw on-screen pause/play button."""
    bx, by, bw, bh = frame_width - 120, 20, 100, 40
    color = (50, 50, 200) if paused else (50, 200, 50)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
    cv2.putText(frame, "PLAY" if paused else "PAUSE",
                (bx + 15, by + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return bx, by, bw, bh

def draw_stop_button(frame):
    """Draw stop button."""
    bx, by, bw, bh = frame_width - 120, 70, 100, 40
    color = (50, 50, 200)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
    cv2.putText(frame, "STOP", (bx + 20, by + 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return bx, by, bw, bh

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks for pause and stop buttons."""
    global paused
    if event == cv2.EVENT_LBUTTONDOWN:
        pause_coords, stop_coords = param
        
        # Check pause button
        pbx, pby, pbw, pbh = pause_coords
        if pbx <= x <= pbx + pbw and pby <= y <= pby + pbh:
            paused = not paused
            print(f"{'Paused' if paused else 'Playing'}")
            return
        
        # Check stop button
        sbx, sby, sbw, sbh = stop_coords
        if sbx <= x <= sbx + sbw and sby <= y <= sby + sbh:
            print("STOP button clicked - Exiting...")
            cv2.destroyAllWindows()
            cap.release()
            exit(0)

##################################################################################################################################

def detect_helmet_detailed(frame, box, track_id):
    """
    Detailed helmet detection with MULTI-RIDER support.
    Checks both driver and passenger areas.
    Returns: (has_helmet, confidence, details_dict)
    """
    x1, y1, x2, y2 = box
    bike_h = y2 - y1
    bike_w = x2 - x1

    # Define TWO head regions: front (driver) and back (passenger)
    # Front rider region
    front_head_y1 = max(0, int(y1 - bike_h * HELMET_REGION_MULTIPLIER))
    front_head_y2 = int(y1 + bike_h * 0.4)
    front_head_x1 = max(0, int(x1 + bike_w * 0.2))  # Front half
    front_head_x2 = min(frame.shape[1], int(x2 + bike_w * 0.3))

    # Back rider region (passenger)
    back_head_y1 = max(0, int(y1 - bike_h * HELMET_REGION_MULTIPLIER))
    back_head_y2 = int(y1 + bike_h * 0.4)
    back_head_x1 = max(0, int(x1 - bike_w * 0.3))  # Back half
    back_head_x2 = int(x1 + bike_w * 0.3)

    # Check BOTH regions
    front_region = frame[front_head_y1:front_head_y2, front_head_x1:front_head_x2]
    back_region = frame[back_head_y1:back_head_y2, back_head_x1:back_head_x2]

    details = {
        'region_size': 0,
        'color_score': 0.0,
        'gloss_score': 0.0,
        'smooth_score': 0.0,
        'shape_score': 0.0,
        'dark_score': 0.0,
        'final_confidence': 0.0,
        'path': 'NONE',
        'is_cloth': False,
        'is_bare': False,
        'decision': 'NO_HELMET',
        'front_helmet': False,
        'back_helmet': False,
        'riders_detected': 1
    }

    # Function to check one region
    def check_region(region):
        if region.size == 0:
            return False, 0.0, {}

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Color detection
        masks = []
        masks.append(cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])))
        masks.append(cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255])))
        masks.append(cv2.inRange(hsv, np.array([20, 100, 120]), np.array([35, 255, 255])))
        masks.append(cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 35, 255])))
        masks.append(cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80])))
        masks.append(cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255])))
        masks.append(cv2.inRange(hsv, np.array([45, 100, 100]), np.array([75, 255, 255])))
        masks.append(cv2.inRange(hsv, np.array([0, 0, 90]), np.array([180, 60, 200])))

        combined_mask = np.zeros_like(masks[0])
        for m in masks:
            combined_mask = cv2.bitwise_or(combined_mask, m)

        helmet_pixels = np.sum(combined_mask > 0)
        total_pixels = region.shape[0] * region.shape[1]
        color_ratio = helmet_pixels / max(total_pixels, 1)
        has_sufficient_size = helmet_pixels > MIN_HELMET_PIXELS

        # Gloss
        _, bright_spots = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(bright_spots > 0) / max(gray.size, 1)
        has_gloss = bright_ratio > 0.02

        # Texture
        mean_f = cv2.blur(gray.astype(float), (5, 5))
        sqr_f = cv2.blur((gray.astype(float))**2, (5, 5))
        variance = np.maximum(sqr_f - mean_f**2, 0)
        avg_text = np.mean(np.sqrt(variance))
        is_smooth = avg_text < PATH2_SMOOTH_THRESHOLD

        # Shape
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_roundness = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 150:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    max_roundness = max(max_roundness, circularity)

        # Darkness
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 70]))
        dark_ratio = np.sum(dark_mask > 0) / max(total_pixels, 1)
        is_very_dark = dark_ratio > PATH2_DARK_THRESHOLD

        # Scores
        color_score = min(1.0, color_ratio / 0.15) if has_sufficient_size else 0.0
        gloss_score = 1.0 if has_gloss else 0.0
        smooth_score = 1.0 if is_smooth else 0.0
        shape_score = min(1.0, max_roundness / 0.5)
        dark_score = 1.0 if is_very_dark else 0.0

        final_conf = (
            color_score * WEIGHT_COLOR +
            gloss_score * WEIGHT_GLOSS +
            smooth_score * WEIGHT_SMOOTH +
            shape_score * WEIGHT_SHAPE +
            dark_score * WEIGHT_DARK
        )

        # Decision paths with balanced requirements
        
        # SAFETY: If shape is VERY poor, likely not a helmet
        # But allow some leeway for distant/angled helmets
        very_poor_shape = shape_score < 0.10  # Lowered from 0.15
        
        if very_poor_shape and not is_very_dark:
            # No shape AND not dark = definitely not helmet
            path1 = False
            path2 = False
            path3 = False
        else:
            path1 = (final_conf > HELMET_CONFIDENCE_THRESHOLD and
                    (gloss_score > PATH1_GLOSS_MIN or smooth_score > PATH1_SMOOTH_MIN) and
                    (color_score > PATH1_COLOR_MIN or shape_score > PATH1_SHAPE_MIN) and
                    has_sufficient_size)

            # Path2: Very dark + smooth + some shape (not perfect shape needed)
            # Also check it's not just dark clothing with high color match
            path2 = (is_very_dark and is_smooth and has_sufficient_size and 
                    shape_score > PATH2_SHAPE_MIN and
                    color_score < 0.40)  # Relaxed from 0.30 - allow some color

            path3 = (shape_score > PATH3_SHAPE_MIN and color_score > PATH3_COLOR_MIN and has_sufficient_size)

        is_cloth = (gloss_score < CLOTH_GLOSS_MAX and smooth_score < CLOTH_SMOOTH_MAX and
                   shape_score < CLOTH_SHAPE_MAX and not is_very_dark and color_score < CLOTH_COLOR_MAX)

        is_bare = (color_score < BARE_COLOR_MAX and shape_score < BARE_SHAPE_MAX and not is_very_dark)

        has_helmet = not is_cloth and not is_bare and (path1 or path2 or path3)

        region_details = {
            'color': color_score,
            'gloss': gloss_score,
            'smooth': smooth_score,
            'shape': shape_score,
            'dark': dark_score,
            'conf': final_conf
        }

        return has_helmet, final_conf, region_details

    # Check front rider (always present)
    front_helmet, front_conf, front_details = check_region(front_region)
    details['front_helmet'] = front_helmet

    # Check back rider (passenger) - only if region has substantial content
    back_helmet = True  # Default: assume no passenger
    back_conf = 0.0
    back_details = {}
    
    # Check if back region has a person (look for high variance = person present)
    if back_region.size > 0:
        gray_back = cv2.cvtColor(back_region, cv2.COLOR_BGR2GRAY)
        back_variance = np.var(gray_back)
        
        # If variance is high, likely a person is there
        if back_variance > 500:  # Threshold for detecting passenger presence
            back_helmet, back_conf, back_details = check_region(back_region)
            details['back_helmet'] = back_helmet
            details['riders_detected'] = 2
        else:
            details['riders_detected'] = 1

    # BOTH riders must have helmets for OK
    overall_has_helmet = front_helmet and back_helmet

    # Use worst confidence
    overall_conf = min(front_conf, back_conf) if details['riders_detected'] == 2 else front_conf

    # Aggregate details (use front rider's details primarily)
    details['color_score'] = front_details.get('color', 0.0)
    details['gloss_score'] = front_details.get('gloss', 0.0)
    details['smooth_score'] = front_details.get('smooth', 0.0)
    details['shape_score'] = front_details.get('shape', 0.0)
    details['dark_score'] = front_details.get('dark', 0.0)
    details['final_confidence'] = overall_conf
    details['decision'] = 'HAS_HELMET' if overall_has_helmet else 'NO_HELMET'

    if not overall_has_helmet:
        if not front_helmet and not back_helmet:
            details['path'] = 'BOTH_NO_HELMET'
        elif not front_helmet:
            details['path'] = 'DRIVER_NO_HELMET'
        else:
            details['path'] = 'PASSENGER_NO_HELMET'
    else:
        details['path'] = 'ALL_OK'

    return overall_has_helmet, overall_conf, details

def save_debug_analysis(frame, box, track_id, details):
    """Save detailed analysis image for debugging."""
    x1, y1, x2, y2 = box
    annotated = frame.copy()

    # Draw box
    color = (0, 255, 0) if details['decision'] == 'HAS_HELMET' else (0, 0, 255)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    # Add detailed text
    y_offset = y1 - 10
    texts = [
        f"ID:{track_id} {details['decision']}",
        f"Path: {details['path']}",
        f"Conf: {details['final_confidence']:.2f}",
        f"Color:{details['color_score']:.2f} Gloss:{details['gloss_score']:.2f}",
        f"Smooth:{details['smooth_score']:.2f} Shape:{details['shape_score']:.2f}",
        f"Dark:{details['dark_score']:.2f}",
        f"Cloth:{details['is_cloth']} Bare:{details['is_bare']}"
    ]

    for txt in texts:
        cv2.putText(annotated, txt, (x1, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_offset -= 15

    timestamp = int(time.time() * 1000)
    filename = f"violations/debug_analysis/ID{track_id}_{timestamp}.jpg"
    cv2.imwrite(filename, annotated)
    print(f"🔍 Debug saved: {filename}")

def save_violation(frame, track_id):
    """Save violation snapshot."""
    timestamp = int(time.time() * 1000)
    cooldown_key = f"no_helmet_{track_id}"

    if cooldown_key not in last_saved_time or \
       (time.time() - last_saved_time[cooldown_key]) > COOLDOWN:
        filename = f"violations/no_helmet/ID{track_id}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        last_saved_time[cooldown_key] = time.time()
        print(f"📸 Violation: {filename}")

##################################################################################################################################
# MAIN LOOP
##################################################################################################################################

cv2.namedWindow("Helmet Detection Focus")
pause_coords = (0, 0, 0, 0)
stop_coords = (0, 0, 0, 0)
frame_count = 0

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

    display_frame = frame.copy()

    # Track motorcycles only
    results = local_model.track(frame, persist=True, classes=TRACK_CLASS, verbose=False)[0]

    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.int().cpu().tolist()
        track_ids = results.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            track_frame_count[track_id] = track_frame_count.get(track_id, 0) + 1

            # Detect helmet with full details
            has_helmet, confidence, details = detect_helmet_detailed(frame, box, track_id)
            is_no_helmet = not has_helmet

            # Track violations over journey
            if track_id not in track_violations:
                track_violations[track_id] = {
                    'no_helmet_count': 0,
                    'total_frames': 0,
                    'details_history': []
                }

            tv = track_violations[track_id]
            tv['total_frames'] += 1
            tv['details_history'].append(details)

            if is_no_helmet:
                tv['no_helmet_count'] += 1

            # Keep only recent history
            if len(tv['details_history']) > 30:
                tv['details_history'] = tv['details_history'][-30:]

            # Decide violation
            has_enough_journey = track_frame_count[track_id] >= MIN_JOURNEY_FRAMES
            total = max(tv['total_frames'], 1)
            no_helmet_ratio = tv['no_helmet_count'] / total

            is_violation = False
            if has_enough_journey and no_helmet_ratio > NO_HELMET_RATIO_THRESHOLD:
                is_violation = True
                violation_ids.add(track_id)
                save_violation(display_frame, track_id)

            # Drawing
            bike_h = y2 - y1
            rider_y_top = max(0, int(y1 - bike_h * HELMET_REGION_MULTIPLIER))

            if is_violation:
                status = f"NO HELMET ({no_helmet_ratio*100:.0f}%)"
                color = (0, 0, 255)
            elif has_enough_journey:
                status = f"OK ({no_helmet_ratio*100:.0f}%)"
                color = (0, 255, 0)
            else:
                status = f"... ({tv['total_frames']}/{MIN_JOURNEY_FRAMES})"
                color = (128, 128, 128)

            cv2.rectangle(display_frame, (x1, rider_y_top), (x2, y2), color, 2)
            cv2.putText(display_frame, status, (x1, rider_y_top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if debug_mode:
                # Show center
                cv2.circle(display_frame, (cx, cy), 5, (255, 0, 255), -1)

                # Detailed debug text
                riders_text = f"{details['riders_detected']} rider(s)"
                helmet_status = f"F:{'✓' if details['front_helmet'] else '✗'}"
                if details['riders_detected'] == 2:
                    helmet_status += f" B:{'✓' if details.get('back_helmet', False) else '✗'}"
                
                dbg_lines = [
                    f"ID:{track_id} {riders_text} {helmet_status}",
                    f"{details['path']}",
                    f"C:{details['final_confidence']:.2f} Col:{details['color_score']:.2f}",
                    f"Gl:{details['gloss_score']:.2f} Sm:{details['smooth_score']:.2f}",
                    f"Sh:{details['shape_score']:.2f} Dk:{details['dark_score']:.2f}",
                    f"NH:{tv['no_helmet_count']}/{total} = {no_helmet_ratio*100:.0f}%"
                ]

                y_pos = y2 + 15
                for line in dbg_lines:
                    cv2.putText(display_frame, line, (x1, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
                    y_pos += 12

    # Dashboard
    cv2.rectangle(display_frame, (10, 10), (450, 120), (0, 0, 0), -1)
    cv2.rectangle(display_frame, (10, 10), (450, 120), (255, 255, 255), 2)

    yo = 30
    cv2.putText(display_frame, f"HELMET DETECTION FOCUS MODE", (20, yo),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    yo += 25
    cv2.putText(display_frame, f"Frame: {frame_count}", (20, yo),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    yo += 20
    cv2.putText(display_frame, f"No Helmet Violations: {len(violation_ids)}", (20, yo),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    yo += 20
    cv2.putText(display_frame, f"Debug: {'ON' if debug_mode else 'OFF'} (press 'd')",
               (20, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Draw buttons
    pause_coords = draw_pause_button(display_frame, paused)
    stop_coords = draw_stop_button(display_frame)
    
    # Set mouse callback with both button coordinates
    cv2.setMouseCallback("Helmet Detection Focus", mouse_callback, (pause_coords, stop_coords))

    cv2.imshow("Helmet Detection Focus", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug: {'ON' if debug_mode else 'OFF'}")
    elif key == ord(' '):
        paused = not paused
        print(f"{'Paused' if paused else 'Playing'}")
    elif key == ord('s'):
        # Save current frame with all annotations
        cv2.imwrite(f"violations/debug_analysis/frame_{frame_count}.jpg", display_frame)
        print(f"Screenshot saved: frame_{frame_count}.jpg")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("HELMET DETECTION RESULTS")
print("="*70)
print(f"Total No Helmet Violations: {len(violation_ids)}")
print("="*70)
print("\nADJUSTMENT GUIDE:")
print("- If missing violations → Lower HELMET_CONFIDENCE_THRESHOLD (try 0.25)")
print("- If too many false positives → Raise HELMET_CONFIDENCE_THRESHOLD (try 0.35)")
print("- Check violations/debug_analysis/ folder for detailed analysis")
print("="*70)
