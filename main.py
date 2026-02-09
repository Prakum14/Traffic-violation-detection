import cv2
import os
import time
from ultralytics import YOLO
from datetime import datetime
import numpy as np

# --- 1. SETUP ---
local_model = YOLO('yolov8n.pt') 

# ===== CURVED DIVIDER CONFIGURATION =====
# Define the road divider as a CURVE using multiple points
# You can add as many points as needed to follow the curve
# Points should go from top to bottom of the frame
# 
# IMPORTANT: Wrong-way detection uses the BOTTOM of the bike (where it touches the road)
# not the center or top (helmet). This ensures accurate side detection.
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
# Add more points for complex curves: (x, y), (x, y), ...

# STEP: Set ONE reference direction
# Watch your video and pick ONE side, note which way traffic flows
REFERENCE_SIDE = "LEFT"           # Which side are you observing?
REFERENCE_DIRECTION = "UP"      # Which way does traffic flow on that side?

# Detection sensitivity
MOVEMENT_THRESHOLD = 12
MIN_TRACK_FRAMES = 12
DIRECTION_HISTORY_SIZE = 25

# Helmet detection sensitivity
HELMET_CONFIDENCE_THRESHOLD = 0.25  # Lower = more sensitive (may have false positives)
HELMET_REGION_MULTIPLIER = 0.8      # How far above bike to look for helmet

# Create violation folders
os.makedirs('violations/no_helmet', exist_ok=True)
os.makedirs('violations/wrong_direction', exist_ok=True)
os.makedirs('violations/combined', exist_ok=True)

track_history = {}
track_directions = {}
violation_ids = {'no_helmet': set(), 'wrong_direction': set(), 'combined': set()}
last_saved_time = {}
COOLDOWN = 3

cap = cv2.VideoCapture("input_videos/sample_traffic.mp4")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("="*70)
print("TRAFFIC VIOLATION DETECTION SYSTEM - REFINED")
print("="*70)
print(f"Video Resolution: {frame_width}x{frame_height}")
print(f"FPS: {fps}")
print(f"Divider Points: {len(DIVIDER_POINTS)} points")
print(f"Reference: {REFERENCE_SIDE} side goes {REFERENCE_DIRECTION}")
print("="*70)
print("Controls:")
print("  'q' - Quit")
print("  'd' - Toggle debug mode")
print("  SPACE - Pause/Play")
print("  Click 'Pause' button on screen")
print("="*70)

debug_mode = False
paused = False
mouse_x, mouse_y = 0, 0

##################################################################################################################################

def interpolate_curve(points, num_interpolated=100):
    """
    Create a smooth curve from given points using interpolation.
    Returns list of (x, y) points along the curve.
    """
    if len(points) < 2:
        return points
    
    # Extract x and y coordinates
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # Create interpolation parameter
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, num_interpolated)
    
    # Interpolate x and y separately
    x_new = np.interp(t_new, t, x_coords)
    y_new = np.interp(t_new, t, y_coords)
    
    # Return as list of tuples
    return list(zip(x_new.astype(int), y_new.astype(int)))

def get_side_of_curve(point, curve_points):
    """
    Determine which side of a curve a point is on.
    Uses the closest segment of the curve.
    """
    px, py = point
    min_dist = float('inf')
    side = "LEFT"
    
    # Find closest segment
    for i in range(len(curve_points) - 1):
        x1, y1 = curve_points[i]
        x2, y2 = curve_points[i + 1]
        
        # Calculate perpendicular distance to segment
        cross_product = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        
        # Distance from point to line segment
        dist = abs(cross_product) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + 0.0001)
        
        if dist < min_dist:
            min_dist = dist
            side = "LEFT" if cross_product > 0 else "RIGHT"
    
    return side

def get_opposite_direction(direction):
    """Get opposite direction."""
    opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    return opposites.get(direction, "UNKNOWN")

def calculate_movement_direction(history):
    """Calculate direction based on position history."""
    if len(history) < MIN_TRACK_FRAMES:
        return "TRACKING", 0.0
    
    sample_size = max(3, len(history) // 3)
    start_positions = history[:sample_size]
    end_positions = history[-sample_size:]
    
    start_x = sum(p[0] for p in start_positions) / len(start_positions)
    start_y = sum(p[1] for p in start_positions) / len(start_positions)
    end_x = sum(p[0] for p in end_positions) / len(end_positions)
    end_y = sum(p[1] for p in end_positions) / len(end_positions)
    
    dx = end_x - start_x
    dy = end_y - start_y
    
    displacement = np.sqrt(dx**2 + dy**2)
    
    if displacement < MOVEMENT_THRESHOLD:
        return "STATIONARY", displacement
    
    if abs(dy) > abs(dx):
        direction = "DOWN" if dy > 0 else "UP"
    else:
        direction = "RIGHT" if dx > 0 else "LEFT"
    
    return direction, displacement

def get_expected_direction(side):
    """Get expected direction for a given side."""
    if side == REFERENCE_SIDE:
        return REFERENCE_DIRECTION
    else:
        return get_opposite_direction(REFERENCE_DIRECTION)

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
    
    confidence = min(1.0, displacement / 50.0)
    is_violation = (detected_direction != expected_direction)
    
    return is_violation, detected_direction, expected_direction, confidence

def detect_helmet_improved(frame, box):
    """
    Improved helmet detection - detects ANY type of helmet including half-face.
    Focus: If there's ANY helmet-like object on head, mark as has_helmet.
    Returns: (has_helmet, confidence)
    """
    x1, y1, x2, y2 = box
    bike_h = y2 - y1
    bike_w = x2 - x1
    
    # Define head/rider region - LARGER area to catch half helmets
    head_y1 = max(0, int(y1 - bike_h * HELMET_REGION_MULTIPLIER))
    head_y2 = int(y1 + bike_h * 0.4)  # Increased from 0.3
    head_x1 = max(0, int(x1 - bike_w * 0.3))  # Increased margin
    head_x2 = min(frame.shape[1], int(x2 + bike_w * 0.3))
    
    head_region = frame[head_y1:head_y2, head_x1:head_x2]
    
    if head_region.size == 0:
        return False, 0.0
    
    # Convert to color spaces
    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
    
    # Method 1: EXPANDED color detection (catches more helmet types)
    masks = []
    
    # Red helmets (broader range, lower saturation threshold)
    masks.append(cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255])))
    masks.append(cv2.inRange(hsv, np.array([165, 40, 40]), np.array([180, 255, 255])))
    
    # Yellow/Orange helmets
    masks.append(cv2.inRange(hsv, np.array([15, 40, 40]), np.array([40, 255, 255])))
    
    # White helmets (broader range)
    masks.append(cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255])))
    
    # Black helmets (broader range)
    masks.append(cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 90])))
    
    # Blue helmets
    masks.append(cv2.inRange(hsv, np.array([90, 40, 40]), np.array([135, 255, 255])))
    
    # Green helmets
    masks.append(cv2.inRange(hsv, np.array([40, 40, 40]), np.array([85, 255, 255])))
    
    # Gray/Silver helmets (common for half-face)
    masks.append(cv2.inRange(hsv, np.array([0, 0, 90]), np.array([180, 60, 200])))
    
    # Brown/Tan (some half helmets)
    masks.append(cv2.inRange(hsv, np.array([5, 30, 30]), np.array([25, 200, 200])))
    
    # Combine all color masks
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Method 2: Edge detection - look for ANY curved edge at top of region
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)  # Lower threshold to catch more
    
    # Check top portion specifically (where helmet would be)
    top_third = edges[:edges.shape[0]//3, :]
    top_edge_density = np.sum(top_third > 0) / (top_third.shape[0] * top_third.shape[1])
    has_top_edge = top_edge_density > 0.05  # Any reasonable edge density
    
    # Method 3: Look for ANY solid object in upper region
    # Apply threshold to find solid regions
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours of solid regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    has_solid_object = False
    largest_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
        if area > 100:  # Any reasonable sized object
            # Check if it's in upper portion of region
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cy = int(M["m01"] / M["m00"])
                if cy < head_region.shape[0] * 0.6:  # In upper 60%
                    has_solid_object = True
    
    # Method 4: Brightness difference (helmet vs hair/skin)
    # Split region into top and bottom
    mid_y = head_region.shape[0] // 2
    top_half = gray[:mid_y, :]
    bottom_half = gray[mid_y:, :]
    
    top_brightness = np.mean(top_half) if top_half.size > 0 else 0
    bottom_brightness = np.mean(bottom_half) if bottom_half.size > 0 else 0
    
    # Helmets are often different brightness than face/hair
    brightness_diff = abs(top_brightness - bottom_brightness)
    has_brightness_contrast = brightness_diff > 15
    
    # Calculate scores
    helmet_pixels = np.sum(combined_mask > 0)
    total_pixels = head_region.shape[0] * head_region.shape[1]
    color_ratio = helmet_pixels / total_pixels if total_pixels > 0 else 0
    
    # Scoring (LENIENT for any helmet type)
    color_score = min(1.0, color_ratio / 0.15)  # Only need 15% color match
    edge_score = 1.0 if has_top_edge else 0.0
    object_score = min(1.0, largest_area / 500.0) if has_solid_object else 0.0
    contrast_score = 1.0 if has_brightness_contrast else 0.3
    
    # LENIENT combination - just need SOME evidence of helmet
    final_confidence = (
        color_score * 0.40 +
        edge_score * 0.25 +
        object_score * 0.25 +
        contrast_score * 0.10
    )
    
    # LENIENT threshold - if ANY indicator suggests helmet, accept it
    has_helmet = (
        final_confidence > HELMET_CONFIDENCE_THRESHOLD or
        (color_score > 0.3) or  # Decent color match
        (has_top_edge and color_score > 0.2) or  # Edge + some color
        (has_solid_object and has_top_edge)  # Solid object with edges
    )
    
    return has_helmet, final_confidence

def save_violation_snapshot(frame, track_id, violations, direction_info=None):
    """Save snapshot ONLY for violations."""
    timestamp = int(time.time() * 1000)
    
    annotated = frame.copy()
    
    if direction_info:
        text = f"ID:{track_id} | Side:{direction_info['side']} | " \
               f"Going:{direction_info['detected']} | Expected:{direction_info['expected']}"
        cv2.putText(annotated, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    for violation_type in violations:
        cooldown_key = f"{violation_type}_{track_id}"
        
        if cooldown_key not in last_saved_time or \
           (time.time() - last_saved_time[cooldown_key]) > COOLDOWN:
            
            if len(violations) > 1:
                folder = 'violations/combined'
                filename = f"{folder}/ID{track_id}_{'_'.join(violations)}_{timestamp}.jpg"
            else:
                folder = f'violations/{violation_type}'
                filename = f"{folder}/ID{track_id}_{timestamp}.jpg"
            
            cv2.imwrite(filename, annotated)
            last_saved_time[cooldown_key] = time.time()
            print(f"📸 Violation saved: {filename}")

def draw_pause_button(frame, paused):
    """Draw on-screen pause/play button."""
    button_x, button_y = frame_width - 120, 20
    button_w, button_h = 100, 40
    
    # Button background
    color = (50, 50, 200) if paused else (50, 200, 50)
    cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), 
                 color, -1)
    cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), 
                 (255, 255, 255), 2)
    
    # Button text
    text = "PLAY" if paused else "PAUSE"
    cv2.putText(frame, text, (button_x + 15, button_y + 28), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return button_x, button_y, button_w, button_h

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks for pause button."""
    global paused, mouse_x, mouse_y
    
    mouse_x, mouse_y = x, y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        button_x, button_y, button_w, button_h = param
        
        # Check if click is within button
        if (button_x <= x <= button_x + button_w and 
            button_y <= y <= button_y + button_h):
            paused = not paused
            print(f"{'Paused' if paused else 'Playing'}")

##################################################################################################################################

# Create interpolated curve for smooth divider
smooth_curve = interpolate_curve(DIVIDER_POINTS, num_interpolated=200)

# Setup mouse callback
cv2.namedWindow("Traffic Monitor")
button_coords = (0, 0, 0, 0)

frame_count = 0

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    
    display_frame = frame.copy()
    
    # --- DRAW THE CURVED DIVIDER ---
    for i in range(len(smooth_curve) - 1):
        cv2.line(display_frame, smooth_curve[i], smooth_curve[i + 1], (0, 255, 255), 3)
    
    # Draw control points (optional, only in debug mode)
    if debug_mode:
        for i, point in enumerate(DIVIDER_POINTS):
            cv2.circle(display_frame, point, 8, (0, 0, 255), -1)
            cv2.putText(display_frame, f"P{i+1}", (point[0] + 10, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # A. TRACK MOTORCYCLES
    results = local_model.track(frame, persist=True, classes=[3], verbose=False)[0]
    
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.int().cpu().tolist()
        track_ids = results.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            # Use LOWER-MIDDLE of bike for direction detection
            # Not the very bottom (which might include wheels/body parts crossing)
            # Not the center (which includes upper body/helmet)
            # Use a point 75% down the bounding box
            bike_track_x = (x1 + x2) // 2
            bike_track_y = int(y1 + (y2 - y1) * 0.75)  # 75% down from top
            
            # Use center for display purposes only
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 

            # Determine side using LOWER-MIDDLE of bike
            side = get_side_of_curve((bike_track_x, bike_track_y), smooth_curve)
            
            # Check direction violation using this tracking point
            direction_violation, detected_dir, expected_dir, dir_confidence = \
                check_direction_violation(track_id, (bike_track_x, bike_track_y), side)

            # Improved helmet detection
            has_helmet, helmet_confidence = detect_helmet_improved(frame, box)
            is_no_helmet = not has_helmet

            # Compile violations
            violations = []
            
            # Only flag if confident enough
            if is_no_helmet and helmet_confidence < (1.0 - HELMET_CONFIDENCE_THRESHOLD):
                violations.append("no_helmet")
                violation_ids['no_helmet'].add(track_id)
            
            if direction_violation and dir_confidence > 0.6:
                violations.append("wrong_direction")
                violation_ids['wrong_direction'].add(track_id)
            
            if len(violations) > 1:
                violation_ids['combined'].add(track_id)

            # Determine box color and label
            bike_h = y2 - y1
            rider_y_top = max(0, int(y1 - (bike_h * HELMET_REGION_MULTIPLIER)))

            if not violations:
                status = "OK"
                color = (0, 255, 0)  # Green
            else:
                status_parts = []
                if "no_helmet" in violations:
                    status_parts.append("No Helmet")
                if "wrong_direction" in violations:
                    status_parts.append("Wrong Way")
                status = " | ".join(status_parts)
                color = (0, 0, 255)  # Red
                
                # Save ONLY violations
                direction_info = {
                    'side': side,
                    'detected': detected_dir,
                    'expected': expected_dir
                }
                save_violation_snapshot(display_frame, track_id, violations, direction_info)

            # Draw bounding box
            cv2.rectangle(display_frame, (x1, rider_y_top), (x2, y2), color, 2)
            cv2.putText(display_frame, status, (x1, rider_y_top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Debug info
            # Debug info
            if debug_mode:
                # Show center point (for reference)
                cv2.circle(display_frame, (cx, cy), 5, (255, 0, 255), -1)
                
                # Show TRACKING point (75% down - what we actually track)
                cv2.circle(display_frame, (bike_track_x, bike_track_y), 8, (0, 255, 0), -1)
                cv2.putText(display_frame, "TRACK", (bike_track_x + 10, bike_track_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                debug_text = f"ID:{track_id}|{side}|{detected_dir}|H:{helmet_confidence:.2f}"
                cv2.putText(display_frame, debug_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Movement trail (using tracking points)
                if track_id in track_history and len(track_history[track_id]) > 1:
                    points = track_history[track_id]
                    for i in range(len(points) - 1):
                        cv2.line(display_frame, points[i], points[i + 1], (0, 255, 0), 2)

    # Dashboard
    dashboard_h = 120
    cv2.rectangle(display_frame, (10, 10), (380, dashboard_h), (0, 0, 0), -1)
    cv2.rectangle(display_frame, (10, 10), (380, dashboard_h), (255, 255, 255), 2)
    
    y_offset = 30
    cv2.putText(display_frame, f"Frame: {frame_count}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(display_frame, f"No Helmet: {len(violation_ids['no_helmet'])}", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    
    y_offset += 25
    cv2.putText(display_frame, f"Wrong Direction: {len(violation_ids['wrong_direction'])}", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    
    y_offset += 25
    cv2.putText(display_frame, f"Combined: {len(violation_ids['combined'])}", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

    # Draw pause/play button
    button_coords = draw_pause_button(display_frame, paused)
    cv2.setMouseCallback("Traffic Monitor", mouse_callback, button_coords)

    cv2.imshow("Traffic Monitor", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    elif key == ord(' '):
        paused = not paused
        print(f"{'Paused' if paused else 'Playing'}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("FINAL STATISTICS")
print("="*70)
print(f"Total No Helmet Violations: {len(violation_ids['no_helmet'])}")
print(f"Total Wrong Direction Violations: {len(violation_ids['wrong_direction'])}")
print(f"Total Combined Violations: {len(violation_ids['combined'])}")
print("="*70)
print(f"\nViolation snapshots saved in 'violations/' folder")
print("Only violations were saved (no OK drivers)")