
import cv2
import os
import time
from ultralytics import YOLO
from datetime import datetime
import numpy as np

# --- 1. SETUP ---
# For helmet detection, we'll use YOLOv8 custom model or built-in person detection
# You can train your own or use a pre-trained one

local_model = YOLO('yolov8n.pt') 

# If you have a custom helmet detection model:
# helmet_model = YOLO('path_to_helmet_model.pt')
# For now, we'll use a simple heuristic based on person detection

# ===== DIRECTION CONFIGURATION =====
# STEP 1: Run calibration tool to get these coordinates
DIVIDER_P1 = (1224, 38)   # Top point - UPDATE THIS
DIVIDER_P2 = (263, 571)   # Bottom point - UPDATE THIS

# STEP 2: Set ONE reference direction
# Watch your video and pick ONE side, note which way traffic flows
# Options: "UP", "DOWN", "LEFT", "RIGHT"
REFERENCE_SIDE = "LEFT"           # Which side are you observing?
REFERENCE_DIRECTION = "UP"       # Which way does traffic flow on that side?

# The system will automatically infer the opposite side should go the opposite direction

# Detection sensitivity
MOVEMENT_THRESHOLD = 10  # Increased from 5 - pixels moved to register as movement
MIN_TRACK_FRAMES = 10    # Increased from 5 - need more frames for reliable detection
DIRECTION_HISTORY_SIZE = 20  # Frames to keep in history

# Create violation folders
os.makedirs('violations/no_helmet', exist_ok=True)
os.makedirs('violations/wrong_direction', exist_ok=True)
os.makedirs('violations/combined', exist_ok=True)
os.makedirs('violations/debug_frames', exist_ok=True)

track_history = {}  # Stores position history
track_directions = {}  # Stores calculated direction for each track
violation_ids = {'no_helmet': set(), 'wrong_direction': set(), 'combined': set()}
last_saved_time = {}
COOLDOWN = 3  # Increased cooldown

cap = cv2.VideoCapture("input_videos/sample_traffic.mp4")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("="*70)
print("TRAFFIC VIOLATION DETECTION SYSTEM V2")
print("="*70)
print(f"Video Resolution: {frame_width}x{frame_height}")
print(f"FPS: {fps}")
print(f"Divider: P1={DIVIDER_P1}, P2={DIVIDER_P2}")
print(f"Reference: {REFERENCE_SIDE} side goes {REFERENCE_DIRECTION}")
print("="*70)
print("Controls:")
print("  'q' - Quit")
print("  'd' - Toggle debug mode")
print("  's' - Save current frame for analysis")
print("="*70)

debug_mode = True

##################################################################################################################################

def get_side_of_line(point, line_p1, line_p2):
    """Determine which side of a line a point is on."""
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return "LEFT" if cross_product > 0 else "RIGHT"

def get_opposite_direction(direction):
    """Get opposite direction."""
    opposites = {
        "UP": "DOWN",
        "DOWN": "UP",
        "LEFT": "RIGHT",
        "RIGHT": "LEFT"
    }
    return opposites.get(direction, "UNKNOWN")

def calculate_movement_direction(history):
    """
    Calculate direction based on position history.
    Uses weighted average giving more importance to recent movements.
    """
    if len(history) < MIN_TRACK_FRAMES:
        return "TRACKING", 0.0  # Not enough data
    
    # Take first 30% and last 30% of history for comparison
    sample_size = max(3, len(history) // 3)
    start_positions = history[:sample_size]
    end_positions = history[-sample_size:]
    
    # Calculate average positions
    start_x = sum(p[0] for p in start_positions) / len(start_positions)
    start_y = sum(p[1] for p in start_positions) / len(start_positions)
    end_x = sum(p[0] for p in end_positions) / len(end_positions)
    end_y = sum(p[1] for p in end_positions) / len(end_positions)
    
    dx = end_x - start_x
    dy = end_y - start_y
    
    # Calculate total displacement
    displacement = np.sqrt(dx**2 + dy**2)
    
    # If movement is too small, consider stationary
    if displacement < MOVEMENT_THRESHOLD:
        return "STATIONARY", displacement
    
    # Determine primary direction based on larger component
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
    """
    Check if vehicle is going in wrong direction.
    Returns: (is_violation, detected_direction, expected_direction, confidence)
    """
    if track_id not in track_history:
        track_history[track_id] = []
    
    track_history[track_id].append(current_pos)
    
    # Keep limited history
    if len(track_history[track_id]) > DIRECTION_HISTORY_SIZE:
        track_history[track_id] = track_history[track_id][-DIRECTION_HISTORY_SIZE:]
    
    # Calculate direction
    detected_direction, displacement = calculate_movement_direction(track_history[track_id])
    
    # Get expected direction for this side
    expected_direction = get_expected_direction(side)
    
    # Store for display
    track_directions[track_id] = {
        'detected': detected_direction,
        'expected': expected_direction,
        'displacement': displacement
    }
    
    # Check for violation only if we have enough data and movement
    if detected_direction in ["TRACKING", "STATIONARY"]:
        return False, detected_direction, expected_direction, 0.0
    
    # Calculate confidence based on displacement
    confidence = min(1.0, displacement / 50.0)  # Normalize to 0-1
    
    is_violation = (detected_direction != expected_direction)
    
    return is_violation, detected_direction, expected_direction, confidence

def detect_helmet_simple(frame, box):
    """
    Simple helmet detection using upper body region analysis.
    This is a basic approach - for better results, use a trained model.
    
    Returns: (has_helmet, confidence)
    """
    x1, y1, x2, y2 = box
    bike_h = y2 - y1
    bike_w = x2 - x1
    
    # Define head region (top portion of the bounding box)
    head_y1 = max(0, int(y1 - bike_h * 0.6))
    head_y2 = int(y1 + bike_h * 0.2)
    head_x1 = max(0, int(x1 - bike_w * 0.1))
    head_x2 = min(frame.shape[1], int(x2 + bike_w * 0.1))
    
    # Extract head region
    head_region = frame[head_y1:head_y2, head_x1:head_x2]
    
    if head_region.size == 0:
        return False, 0.0
    
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for common helmet colors
    # Red helmets
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Yellow helmets
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # White helmets
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    # Black helmets
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    
    # Create masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Combine masks
    combined_mask = mask_red1 | mask_red2 | mask_yellow | mask_white | mask_black
    
    # Calculate percentage of helmet-colored pixels
    helmet_pixels = np.sum(combined_mask > 0)
    total_pixels = head_region.shape[0] * head_region.shape[1]
    helmet_ratio = helmet_pixels / total_pixels if total_pixels > 0 else 0
    
    # If more than 15% of head region has helmet colors, consider it a helmet
    has_helmet = helmet_ratio > 0.15
    confidence = min(1.0, helmet_ratio / 0.3)  # Normalize
    
    return has_helmet, confidence

def save_violation_snapshot(frame, track_id, violations, direction_info=None):
    """Save snapshot to appropriate folder(s)."""
    timestamp = int(time.time() * 1000)
    
    # Create annotated frame
    annotated = frame.copy()
    
    if direction_info:
        text = f"ID:{track_id} | Side:{direction_info['side']} | " \
               f"Going:{direction_info['detected']} | Expected:{direction_info['expected']}"
        cv2.putText(annotated, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
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
            print(f"📸 Saved: {filename}")

##################################################################################################################################

frame_count = 0
paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    
    display_frame = frame.copy()
    
    # --- DRAW THE DIVIDER ---
    cv2.line(display_frame, DIVIDER_P1, DIVIDER_P2, (0, 255, 255), 3)
    
    # Add direction indicators
    mid_point = ((DIVIDER_P1[0] + DIVIDER_P2[0]) // 2, 
                 (DIVIDER_P1[1] + DIVIDER_P2[1]) // 2)
    
    # Determine arrow positions based on line orientation
    dx = DIVIDER_P2[0] - DIVIDER_P1[0]
    dy = DIVIDER_P2[1] - DIVIDER_P1[1]
    length = np.sqrt(dx*dx + dy*dy)
    
    if length > 0:
        # Perpendicular offset
        perp_dx = -dy / length * 100
        perp_dy = dx / length * 100
        
        # LEFT side
        left_pos = (int(mid_point[0] + perp_dx), int(mid_point[1] + perp_dy))
        left_dir = REFERENCE_DIRECTION if REFERENCE_SIDE == "LEFT" else get_opposite_direction(REFERENCE_DIRECTION)
        
        # Draw arrow for left side
        if left_dir == "UP":
            cv2.arrowedLine(display_frame, (left_pos[0], left_pos[1] + 40), 
                          (left_pos[0], left_pos[1] - 40), (255, 100, 100), 3, tipLength=0.3)
        elif left_dir == "DOWN":
            cv2.arrowedLine(display_frame, (left_pos[0], left_pos[1] - 40), 
                          (left_pos[0], left_pos[1] + 40), (255, 100, 100), 3, tipLength=0.3)
        
        cv2.putText(display_frame, f"LEFT: {left_dir}", (left_pos[0] - 50, left_pos[1] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        
        # RIGHT side
        right_pos = (int(mid_point[0] - perp_dx), int(mid_point[1] - perp_dy))
        right_dir = REFERENCE_DIRECTION if REFERENCE_SIDE == "RIGHT" else get_opposite_direction(REFERENCE_DIRECTION)
        
        # Draw arrow for right side
        if right_dir == "UP":
            cv2.arrowedLine(display_frame, (right_pos[0], right_pos[1] + 40), 
                          (right_pos[0], right_pos[1] - 40), (100, 100, 255), 3, tipLength=0.3)
        elif right_dir == "DOWN":
            cv2.arrowedLine(display_frame, (right_pos[0], right_pos[1] - 40), 
                          (right_pos[0], right_pos[1] + 40), (100, 100, 255), 3, tipLength=0.3)
        
        cv2.putText(display_frame, f"RIGHT: {right_dir}", (right_pos[0] - 50, right_pos[1] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

    # A. TRACK MOTORCYCLES
    results = local_model.track(frame, persist=True, classes=[3], verbose=False)[0]
    
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.int().cpu().tolist()
        track_ids = results.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 

            # Determine side
            side = get_side_of_line((cx, cy), DIVIDER_P1, DIVIDER_P2)
            
            # Check direction violation
            direction_violation, detected_dir, expected_dir, dir_confidence = \
                check_direction_violation(track_id, (cx, cy), side)

            # Helmet detection (using simple method - replace with trained model for better results)
            has_helmet, helmet_confidence = detect_helmet_simple(frame, box)
            is_no_helmet = not has_helmet

            # Compile violations
            violations = []
            
            if is_no_helmet and helmet_confidence > 0.3:  # Only flag if reasonably confident
                violations.append("no_helmet")
                violation_ids['no_helmet'].add(track_id)
            
            # Only flag direction violation if we're confident
            if direction_violation and dir_confidence > 0.5:
                violations.append("wrong_direction")
                violation_ids['wrong_direction'].add(track_id)
            
            if len(violations) > 1:
                violation_ids['combined'].add(track_id)

            # Determine box color and label
            bike_h = y2 - y1
            rider_y_top = max(0, int(y1 - (bike_h * 0.6)))

            if not violations:
                status = "OK"
                color = (0, 255, 0)  # Green
            else:
                status_parts = []
                if "no_helmet" in violations:
                    status_parts.append(f"No Helmet({helmet_confidence:.1f})")
                if "wrong_direction" in violations:
                    status_parts.append(f"Wrong Way({dir_confidence:.1f})")
                status = " | ".join(status_parts)
                color = (0, 0, 255)  # Red
                
                # Save snapshot
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
            if debug_mode:
                # Center point
                cv2.circle(display_frame, (cx, cy), 5, (255, 0, 255), -1)
                
                # Debug text
                debug_text = f"ID:{track_id} | {side} | {detected_dir}"
                cv2.putText(display_frame, debug_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Movement trail
                if track_id in track_history and len(track_history[track_id]) > 1:
                    points = track_history[track_id]
                    for i in range(len(points) - 1):
                        cv2.line(display_frame, points[i], points[i + 1], (255, 0, 255), 2)

    # Dashboard
    dashboard_h = 140
    cv2.rectangle(display_frame, (10, 10), (400, dashboard_h), (0, 0, 0), -1)
    cv2.rectangle(display_frame, (10, 10), (400, dashboard_h), (255, 255, 255), 2)
    
    y_offset = 30
    cv2.putText(display_frame, f"Frame: {frame_count}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(display_frame, f"No Helmet Violations: {len(violation_ids['no_helmet'])}", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    
    y_offset += 25
    cv2.putText(display_frame, f"Wrong Direction: {len(violation_ids['wrong_direction'])}", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    
    y_offset += 25
    cv2.putText(display_frame, f"Combined: {len(violation_ids['combined'])}", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    
    y_offset += 25
    debug_status = "ON" if debug_mode else "OFF"
    cv2.putText(display_frame, f"Debug: {debug_status} (press 'd')", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("Traffic AI Monitor V2", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    elif key == ord('s'):
        cv2.imwrite(f"violations/debug_frames/frame_{frame_count}.jpg", display_frame)
        print(f"Saved debug frame: frame_{frame_count}.jpg")
    elif key == ord(' '):  # Space to pause
        paused = not paused
        print(f"{'Paused' if paused else 'Resumed'}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("FINAL STATISTICS")
print("="*70)
print(f"Total No Helmet Violations: {len(violation_ids['no_helmet'])}")
print(f"Total Wrong Direction Violations: {len(violation_ids['wrong_direction'])}")
print(f"Total Combined Violations: {len(violation_ids['combined'])}")
print("="*70)