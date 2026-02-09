import cv2
import os
import time
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
from datetime import datetime

# --- 1. SETUP ---
API_KEY = "4JxwrQcHLhZu0YOhEB50" 
MODEL_ID = "helmet-detection-76mok/5"

local_model = YOLO('yolov8n.pt') 
client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=API_KEY)

# Keep everything above this line as it is and only change Model_ID if required

# DEFINE THE WHITE STRIP (Adjust these to match your road's divider)
# Format: (x, y). Point 1 is far (top), Point 2 is near (bottom).
DIVIDER_P1 = (1200, 20) 
DIVIDER_P2 = (-10, 800)

os.makedirs('violations', exist_ok=True)
track_history = {} 
violation_ids = set() # Unique set to count total violations
last_saved_time = 0
COOLDOWN = 2 # Seconds between saving snapshots

# Helper function to check which side of the line a point is on
def get_lane_side(x, y, p1, p2):
    # Cross product formula
    val = (p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0])
    return "LEFT" if val < 0 else "RIGHT"

cap = cv2.VideoCapture("input_videos/sample_traffic.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # DRAW THE WHITE STRIP DIVIDER (Yellow for visibility)
    cv2.line(frame, DIVIDER_P1, DIVIDER_P2, (0, 255, 255), 3)
    cv2.putText(frame, "CENTER DIVIDER", (DIVIDER_P1[0]-100, DIVIDER_P1[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # A. TRACK MOTORCYCLES
    results = local_model.track(frame, persist=True, classes=[3], verbose=False)[0]
    
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.int().cpu().tolist()
        track_ids = results.boxes.id.int().cpu().tolist()

        # B. HELMET CHECK
        api_result = client.infer(frame, model_id=MODEL_ID)
        api_preds = api_result.get("predictions", [])
        has_helmet = any(p['class'].lower() == "helmet" and p['confidence'] > 0.60 for p in api_preds)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # Center point
            
            # --- LANE & DIRECTION LOGIC ---
            lane = get_lane_side(cx, cy, DIVIDER_P1, DIVIDER_P2)
            direction_violation = False
            
            if track_id in track_history:
                prev_y = track_history[track_id]
                movement = cy - prev_y # (+) = Down, (-) = Up

                # Apply logic based on Lane
                if lane == "LEFT" and movement > 1: # Moving DOWN in Left lane
                    direction_violation = True
                elif lane == "RIGHT" and movement < -1: # Moving UP in Right lane
                    direction_violation = True

            track_history[track_id] = cy

            # --- VISUALS ---
            bike_h = y2 - y1
            rider_y = max(0, int(y1 - (bike_h * 0.6)))
            no_helmet = not has_helmet
            
            if no_helmet or direction_violation:
                color = (0, 0, 255)
                status = "VIOLATION"
                violation_ids.add(track_id) # Count unique IDs
                
                if (time.time() - last_saved_time) > 2:
                    cv2.imwrite(f"violations/ID_{track_id}_{int(time.time())}.jpg", frame)
                    last_saved_time = time.time()
            else:
                color = (0, 255, 0)
                status = "OK"

            cv2.rectangle(frame, (x1, rider_y), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id} {lane} {status}", (x1, rider_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # C. DISPLAY COUNTER
    cv2.rectangle(frame, (20, 20), (250, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"TOTAL VIOLATIONS: {len(violation_ids)}", (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Smart Lane Enforcement AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()