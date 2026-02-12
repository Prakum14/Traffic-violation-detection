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
# IMPORTANT: Wrong-way detection uses 70% down the bike (stable tracking point)
# not the center or bottom edge. This prevents false positives from body parts crossing.
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
MIN_TRACK_FRAMES = 5
DIRECTION_HISTORY_SIZE = 25
DIVIDER_BUFFER_ZONE = 50  # Pixels - bikes within this distance to divider are not flagged

# Helmet detection sensitivity - STRICTER now
HELMET_CONFIDENCE_THRESHOLD = 0.38  # Increased to reject cloth/caps
HELMET_REGION_MULTIPLIER = 0.8      # How far above bike to look for helmet

# Tracking point adjustment
BIKE_TRACKING_PERCENTAGE = 0.40  # 40% down the bounding box (adjust if needed: 0.65, 0.75)

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
print("TRAFFIC VIOLATION DETECTION SYSTEM - FINAL VERSION")
print("="*70)
print(f"Video Resolution: {frame_width}x{frame_height}")
print(f"FPS: {fps}")
print(f"Divider Points: {len(DIVIDER_POINTS)} points")
print(f"Reference: {REFERENCE_SIDE} side goes {REFERENCE_DIRECTION}")
print(f"Buffer Zone: {DIVIDER_BUFFER_ZONE} pixels")
print(f"Helmet Threshold: {HELMET_CONFIDENCE_THRESHOLD}")
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
# ALL FUNCTION DEFINITIONS MUST BE HERE (BEFORE MAIN LOOP)
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

def get_distance_to_curve(point, curve_points):
    """
    Calculate minimum distance from point to curve.
    Used for buffer zone detection.
    """
    px, py = point
    min_distance = float('inf')
    
    for curve_point in curve_points:
        dist = np.sqrt((px - curve_point[0])**2 + (py - curve_point[1])**2)
        min_distance = min(min_distance, dist)
    
    return min_distance

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
    STRICT helmet detection - only accepts HARD helmets, rejects cloth/caps.
    Looks for: glossy surface, solid structure, helmet-specific colors.
    Returns: (has_helmet, confidence)
    """
    x1, y1, x2, y2 = box
    bike_h = y2 - y1
    bike_w = x2 - x1
    
    # Define head region
    head_y1 = max(0, int(y1 - bike_h * HELMET_REGION_MULTIPLIER))
    head_y2 = int(y1 + bike_h * 0.4)
    head_x1 = max(0, int(x1 - bike_w * 0.3))
    head_x2 = min(frame.shape[1], int(x2 + bike_w * 0.3))
    
    head_region = frame[head_y1:head_y2, head_x1:head_x2]
    
    if head_region.size == 0:
        return False, 0.0
    
    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
    
    # Method 1: STRICT color detection - ONLY bright/saturated helmet colors
    # Excludes dull cloth colors
    masks = []
    
    # Bright Red helmets (HIGH saturation only - no dull red cloth)
    masks.append(cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])))
    masks.append(cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255])))
    
    # Bright Yellow/Orange helmets (HIGH saturation - no beige/tan cloth)
    masks.append(cv2.inRange(hsv, np.array([20, 100, 120]), np.array([35, 255, 255])))
    
    # GLOSSY White helmets (VERY bright, low saturation - not dull white cloth)
    masks.append(cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 35, 255])))
    
    # GLOSSY Black helmets (VERY dark, very low saturation - not black cloth)
    masks.append(cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 30, 60])))
    
    # Bright Blue helmets (HIGH saturation only)
    masks.append(cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255])))
    
    # Bright Green helmets (HIGH saturation only)
    masks.append(cv2.inRange(hsv, np.array([45, 100, 100]), np.array([75, 255, 255])))
    
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Method 2: DETECT GLOSSY/REFLECTIVE SURFACE (helmets reflect light, cloth doesn't)
    # Look for high-intensity spots (specular reflections)
    _, bright_spots = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_ratio = np.sum(bright_spots > 0) / (gray.shape[0] * gray.shape[1])
    has_gloss = bright_ratio > 0.01  # At least 1% bright reflective pixels
    
    # Method 3: TEXTURE ANALYSIS (helmets are smooth, cloth is textured)
    # Calculate local standard deviation
    kernel_size = 5
    mean_filtered = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
    sqr_mean_filtered = cv2.blur((gray.astype(float))**2, (kernel_size, kernel_size))
    variance = sqr_mean_filtered - mean_filtered**2
    variance = np.maximum(variance, 0)  # Ensure non-negative
    std_dev = np.sqrt(variance)
    
    # Helmets have LOW texture variance (smooth), cloth has HIGH variance
    avg_texture = np.mean(std_dev)
    is_smooth = avg_texture < 20  # Smooth surface threshold
    
    # Method 4: SHAPE - Look for ROUNDED/DOME shape (helmet characteristic)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    has_dome_shape = False
    max_roundness = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:  # Minimum helmet size
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # Circularity: 1.0 = perfect circle, helmets are ~0.6-0.9
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                max_roundness = max(max_roundness, circularity)
                if circularity > 0.55:  # Helmet-like roundness
                    # Also check if it's in upper portion
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cy = int(M["m01"] / M["m00"])
                        if cy < head_region.shape[0] * 0.5:  # Top half
                            has_dome_shape = True
    
    # Method 5: SIZE CHECK - Helmets have minimum size
    helmet_pixels = np.sum(combined_mask > 0)
    total_pixels = head_region.shape[0] * head_region.shape[1]
    color_ratio = helmet_pixels / total_pixels if total_pixels > 0 else 0
    has_sufficient_size = helmet_pixels > 300  # Minimum pixels for a helmet
    
    # Calculate individual scores
    color_score = min(1.0, color_ratio / 0.15) if has_sufficient_size else 0.0
    gloss_score = 1.0 if has_gloss else 0.0
    smooth_score = 1.0 if is_smooth else 0.0
    shape_score = min(1.0, max_roundness / 0.6) if has_dome_shape else 0.0
    
    # STRICT SCORING - Need MULTIPLE positive indicators
    # Cloth/caps will fail because they lack gloss OR smooth surface
    final_confidence = (
        color_score * 0.30 +
        gloss_score * 0.25 +    # IMPORTANT: Cloth has no gloss
        smooth_score * 0.25 +   # IMPORTANT: Cloth is textured
        shape_score * 0.20
    )
    
    # STRICT REQUIREMENTS - Must satisfy multiple criteria
    has_helmet = (
        final_confidence > HELMET_CONFIDENCE_THRESHOLD and
        (gloss_score > 0.5 or smooth_score > 0.5) and  # Must be glossy OR smooth
        (color_score > 0.3 or shape_score > 0.4) and   # Must have color OR shape
        has_sufficient_size  # Must be big enough
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
# MAIN LOOP STARTS HERE
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
    
    # Draw buffer zone (optional, only in debug mode)
    if debug_mode:
        # Draw divider control points
        for i, point in enumerate(DIVIDER_POINTS):
            cv2.circle(display_frame, point, 8, (0, 0, 255), -1)
            cv2.putText(display_frame, f"P{i+1}", (point[0] + 10, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw buffer zone as semi-transparent area
        overlay = display_frame.copy()
        for i in range(len(smooth_curve) - 1):
            # Create buffer on both sides of divider
            x1, y1 = smooth_curve[i]
            x2, y2 = smooth_curve[i + 1]
            
            # Calculate perpendicular direction
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                perp_dx = -dy / length * DIVIDER_BUFFER_ZONE
                perp_dy = dx / length * DIVIDER_BUFFER_ZONE
                
                # Left side buffer
                left1 = (int(x1 + perp_dx), int(y1 + perp_dy))
                left2 = (int(x2 + perp_dx), int(y2 + perp_dy))
                cv2.line(overlay, left1, left2, (100, 100, 0), 2)
                
                # Right side buffer
                right1 = (int(x1 - perp_dx), int(y1 - perp_dy))
                right2 = (int(x2 - perp_dx), int(y2 - perp_dy))
                cv2.line(overlay, right1, right2, (100, 100, 0), 2)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)

    # A. TRACK MOTORCYCLES
    results = local_model.track(frame, persist=True, classes=[3], verbose=False)[0]
    
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.int().cpu().tolist()
        track_ids = results.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            # Use LOWER-MIDDLE of bike for direction detection
            bike_track_x = (x1 + x2) // 2
            bike_track_y = int(y1 + (y2 - y1) * BIKE_TRACKING_PERCENTAGE)
            
            # Use center for display purposes only
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 

            # Determine side using LOWER-MIDDLE of bike
            side = get_side_of_curve((bike_track_x, bike_track_y), smooth_curve)
            
            # OPTIONAL: Uncomment these lines if your sides are inverted
            #if side == "LEFT":
             #    side = "RIGHT"
            #elif side == "RIGHT":
             #    side = "LEFT"
            
            # Calculate distance to divider for buffer zone
            distance_to_divider = get_distance_to_curve((bike_track_x, bike_track_y), smooth_curve)
            is_near_divider = distance_to_divider < DIVIDER_BUFFER_ZONE
            
            # Check direction violation (skip if in buffer zone)
            if not is_near_divider:
                direction_violation, detected_dir, expected_dir, dir_confidence = \
                    check_direction_violation(track_id, (bike_track_x, bike_track_y), side)
            else:
                # Skip violation check for bikes near divider
                direction_violation = False
                detected_dir = "BUFFER"
                expected_dir = get_expected_direction(side)
                dir_confidence = 0.0

            # Strict helmet detection
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
            if debug_mode:
                # Show center point (for reference)
                cv2.circle(display_frame, (cx, cy), 5, (255, 0, 255), -1)
                
                # Show TRACKING point (what we actually track)
                track_color = (255, 165, 0) if is_near_divider else (0, 255, 0)  # Orange if in buffer
                cv2.circle(display_frame, (bike_track_x, bike_track_y), 8, track_color, -1)
                
                track_label = "BUFFER" if is_near_divider else "TRACK"
                cv2.putText(display_frame, track_label, (bike_track_x + 10, bike_track_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, track_color, 1)
                
                debug_text = f"ID:{track_id}|{side}|{detected_dir}|H:{helmet_confidence:.2f}|D:{int(distance_to_divider)}"
                cv2.putText(display_frame, debug_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Movement trail (using tracking points)
                if track_id in track_history and len(track_history[track_id]) > 1:
                    points = track_history[track_id]
                    for i in range(len(points) - 1):
                        cv2.line(display_frame, points[i], points[i + 1], (0, 255, 0), 2)

    # Dashboard
    dashboard_h = 140
    cv2.rectangle(display_frame, (10, 10), (400, dashboard_h), (0, 0, 0), -1)
    cv2.rectangle(display_frame, (10, 10), (400, dashboard_h), (255, 255, 255), 2)
    
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
    
    y_offset += 25
    debug_status = "ON" if debug_mode else "OFF"
    cv2.putText(display_frame, f"Debug: {debug_status} (press 'd')", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

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
