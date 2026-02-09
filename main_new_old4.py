import cv2
import os
import time
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
from datetime import datetime

# --- 1. SETUP ---
API_KEY = "4JxwrQcHLhZu0YOhEB50" 
MODEL_ID = "helmet-detection-76mok/3"

local_model = YOLO('yolov8n.pt') 
client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=API_KEY)

# ===== DIVIDER CONFIGURATION =====
# IMPORTANT: Adjust these coordinates based on your video
# The divider should separate the two traffic lanes
# Format: (x, y) coordinates
# You may need to run the video once and note where the divider should be
DIVIDER_P1 = (1224, 38)   # Top point of divider (adjust based on your video)
DIVIDER_P2 = (263, 571)   # Bottom point of divider (adjust based on your video)

# Direction configuration
# Set the expected direction for each side
# "UP" means vehicle should move upward in frame (y decreasing)
# "DOWN" means vehicle should move downward in frame (y increasing)
LEFT_SIDE_DIRECTION = "DOWN"   # Expected direction for left side of divider
RIGHT_SIDE_DIRECTION = "UP"    # Expected direction for right side of divider

# Detection sensitivity
MOVEMENT_THRESHOLD = 5  # Minimum pixel movement to detect direction
MIN_TRACK_FRAMES = 5    # Minimum frames before checking direction

# Create violation folders
os.makedirs('violations/no_helmet', exist_ok=True)
os.makedirs('violations/wrong_direction', exist_ok=True)
os.makedirs('violations/combined', exist_ok=True)

track_history = {}  # Stores position history: {track_id: [(x, y), ...]}
violation_ids = {'no_helmet': set(), 'wrong_direction': set(), 'combined': set()}
last_saved_time = {}  # Track last save time per violation type per ID
COOLDOWN = 2 

cap = cv2.VideoCapture("input_videos/sample_traffic.mp4")

# Get video properties for better coordinate suggestions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("="*60)
print("Traffic Violation Detection System")
print("="*60)
print(f"Video Resolution: {frame_width}x{frame_height}")
print(f"FPS: {fps}")
print(f"Current Divider: P1={DIVIDER_P1}, P2={DIVIDER_P2}")
print(f"Left Side Expected: {LEFT_SIDE_DIRECTION}")
print(f"Right Side Expected: {RIGHT_SIDE_DIRECTION}")
print("="*60)
print("Press 'q' to quit, 'd' to toggle debug view")
print("="*60)

debug_mode = True

##################################################################################################################################

def get_side_of_line(point, line_p1, line_p2):
    """
    Determine which side of a line a point is on using cross product.
    Returns: True if on left side, False if on right side
    """
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    
    cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return cross_product > 0

def get_movement_direction(history):
    """
    Determine the overall movement direction from position history.
    Returns: "UP", "DOWN", "LEFT", "RIGHT", or "STATIONARY"
    """
    if len(history) < 2:
        return "UNKNOWN"
    
    # Compare first few and last few positions
    start_positions = history[:min(3, len(history)//2)]
    end_positions = history[-min(3, len(history)//2):]
    
    start_y = sum(p[1] for p in start_positions) / len(start_positions)
    end_y = sum(p[1] for p in end_positions) / len(end_positions)
    
    start_x = sum(p[0] for p in start_positions) / len(start_positions)
    end_x = sum(p[0] for p in end_positions) / len(end_positions)
    
    dy = end_y - start_y
    dx = end_x - start_x
    
    # Check if movement is significant
    if abs(dy) < MOVEMENT_THRESHOLD and abs(dx) < MOVEMENT_THRESHOLD:
        return "STATIONARY"
    
    # Determine primary direction
    if abs(dy) > abs(dx):
        return "DOWN" if dy > 0 else "UP"
    else:
        return "RIGHT" if dx > 0 else "LEFT"

def check_direction_violation(track_id, current_pos, is_on_left):
    """
    Check if vehicle is going in wrong direction.
    Returns: (is_violation, movement_direction)
    """
    if track_id not in track_history:
        track_history[track_id] = []
    
    track_history[track_id].append(current_pos)
    
    # Keep only recent history (last 15 frames)
    if len(track_history[track_id]) > 15:
        track_history[track_id] = track_history[track_id][-15:]
    
    # Need minimum frames to determine direction
    if len(track_history[track_id]) < MIN_TRACK_FRAMES:
        return False, "UNKNOWN"
    
    movement_dir = get_movement_direction(track_history[track_id])
    
    if movement_dir in ["UNKNOWN", "STATIONARY"]:
        return False, movement_dir
    
    # Check if movement matches expected direction
    expected_dir = LEFT_SIDE_DIRECTION if is_on_left else RIGHT_SIDE_DIRECTION
    
    is_violation = (movement_dir != expected_dir)
    
    return is_violation, movement_dir

def save_violation_snapshot(frame, track_id, violations):
    """
    Save snapshot to appropriate folder(s) based on violation type(s).
    """
    timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
    
    for violation_type in violations:
        # Check cooldown for this specific violation type and ID
        cooldown_key = f"{violation_type}_{track_id}"
        
        if cooldown_key not in last_saved_time or \
           (time.time() - last_saved_time[cooldown_key]) > COOLDOWN:
            
            if len(violations) > 1:
                folder = 'violations/combined'
                filename = f"{folder}/ID{track_id}_{'_'.join(violations)}_{timestamp}.jpg"
            else:
                folder = f'violations/{violation_type}'
                filename = f"{folder}/ID{track_id}_{timestamp}.jpg"
            
            cv2.imwrite(filename, frame)
            last_saved_time[cooldown_key] = time.time()
            print(f"📸 Saved: {filename}")

##################################################################################################################################

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # --- DRAW THE DIVIDER ---
    cv2.line(frame, DIVIDER_P1, DIVIDER_P2, (0, 255, 255), 3)
    
    # Add arrows to show expected directions
    mid_point = ((DIVIDER_P1[0] + DIVIDER_P2[0]) // 2, (DIVIDER_P1[1] + DIVIDER_P2[1]) // 2)
    
    # Left side arrow
    left_arrow_x = mid_point[0] - 80
    if LEFT_SIDE_DIRECTION == "UP":
        cv2.arrowedLine(frame, (left_arrow_x, mid_point[1] + 30), 
                       (left_arrow_x, mid_point[1] - 30), (0, 255, 0), 2, tipLength=0.3)
    else:
        cv2.arrowedLine(frame, (left_arrow_x, mid_point[1] - 30), 
                       (left_arrow_x, mid_point[1] + 30), (0, 255, 0), 2, tipLength=0.3)
    
    # Right side arrow
    right_arrow_x = mid_point[0] + 80
    if RIGHT_SIDE_DIRECTION == "UP":
        cv2.arrowedLine(frame, (right_arrow_x, mid_point[1] + 30), 
                       (right_arrow_x, mid_point[1] - 30), (0, 255, 0), 2, tipLength=0.3)
    else:
        cv2.arrowedLine(frame, (right_arrow_x, mid_point[1] - 30), 
                       (right_arrow_x, mid_point[1] + 30), (0, 255, 0), 2, tipLength=0.3)

    # A. TRACK MOTORCYCLES
    results = local_model.track(frame, persist=True, classes=[3], verbose=False)[0]
    
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.int().cpu().tolist()
        track_ids = results.boxes.id.int().cpu().tolist()

        # B. GET API DETECTIONS (Check the whole frame for helmets)
        api_result = client.infer(frame, model_id=MODEL_ID)
        api_preds = api_result.get("predictions", [])

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 

            # C. DETERMINE SIDE OF DIVIDER
            is_on_left = get_side_of_line((cx, cy), DIVIDER_P1, DIVIDER_P2)
            
            # D. CHECK DIRECTION VIOLATION
            direction_violation, movement_dir = check_direction_violation(
                track_id, (cx, cy), is_on_left
            )

            # E. HELMET DETECTION
            rider_has_helmet = False
            for p in api_preds:
                px, py = int(p['x']), int(p['y'])
                label = p['class'].lower()
                conf = p['confidence']

                if ("helmet" in label or "safety" in label) and conf > 0.25:
                    if (x1 - 40) <= px <= (x2 + 40):
                        if py < y2 - ((y2 - y1) * 0.3): 
                            rider_has_helmet = True
                            if debug_mode:
                                # Draw helmet detection for debugging
                                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
                            break
            
            # F. COMPILE VIOLATIONS
            is_no_helmet = not rider_has_helmet
            violations = []
            
            if is_no_helmet: 
                violations.append("no_helmet")
                violation_ids['no_helmet'].add(track_id)
            
            if direction_violation: 
                violations.append("wrong_direction")
                violation_ids['wrong_direction'].add(track_id)
            
            if len(violations) > 1:
                violation_ids['combined'].add(track_id)

            # G. DETERMINE BOX COLOR AND LABEL
            bike_h = y2 - y1
            rider_y_top = max(0, int(y1 - (bike_h * 0.6)))

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
                
                # Save snapshot
                save_violation_snapshot(frame, track_id, violations)

            # H. DRAW BOUNDING BOX AND LABELS
            cv2.rectangle(frame, (x1, rider_y_top), (x2, y2), color, 2)
            
            # Main status label
            cv2.putText(frame, status, (x1, rider_y_top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Debug info
            if debug_mode:
                # Draw center point
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
                
                # Show side and movement
                side_text = "LEFT" if is_on_left else "RIGHT"
                debug_text = f"ID:{track_id} | {side_text} | {movement_dir}"
                cv2.putText(frame, debug_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Draw movement trail
                if track_id in track_history and len(track_history[track_id]) > 1:
                    points = track_history[track_id]
                    for i in range(len(points) - 1):
                        cv2.line(frame, points[i], points[i + 1], (255, 0, 255), 1)

    # I. DASHBOARD
    dashboard_h = 120
    cv2.rectangle(frame, (10, 10), (350, dashboard_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (350, dashboard_h), (255, 255, 255), 2)
    
    y_offset = 30
    cv2.putText(frame, f"Frame: {frame_count}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(frame, f"No Helmet: {len(violation_ids['no_helmet'])}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    
    y_offset += 25
    cv2.putText(frame, f"Wrong Direction: {len(violation_ids['wrong_direction'])}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
    
    y_offset += 25
    cv2.putText(frame, f"Combined: {len(violation_ids['combined'])}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

    cv2.imshow("Traffic AI Monitor", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("FINAL STATISTICS")
print("="*60)
print(f"Total No Helmet Violations: {len(violation_ids['no_helmet'])}")
print(f"Total Wrong Direction Violations: {len(violation_ids['wrong_direction'])}")
print(f"Total Combined Violations: {len(violation_ids['combined'])}")
print("="*60)
