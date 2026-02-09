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

# 3. VIDEO SOURCE
# Use 0 for webcam or path to your mp4 file
cap = cv2.VideoCapture("input_videos/sample_traffic.mp4")

print("System Initialized. Monitoring Dual-Lane Traffic...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape

    # Draw the Divider Line (Visual Reference)
    cv2.line(frame, DIVIDER_P1, DIVIDER_P2, (0, 255, 255), 2)
    cv2.putText(frame, "CENTER DIVIDER", (DIVIDER_P1[0]-100, DIVIDER_P1[1]-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # A. TRACK MOTORCYCLES (Using YOLOv8 Tracker)
    # class 3 is motorcycle
    results = local_model.track(frame, persist=True, classes=[3], verbose=False)[0]
    
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.int().cpu().tolist()
        track_ids = results.boxes.id.int().cpu().tolist()

        # B. HELMET INTELLIGENCE (Cloud API Call)
        api_result = client.infer(frame, model_id=MODEL_ID)
        api_preds = api_result.get("predictions", [])
        
        # Check if a high-confidence helmet exists anywhere in the frame
        helmet_detected = any(p['class'].lower() == "helmet" and p['confidence'] > 0.40 for p in api_preds)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # Center point
            
            # C. SIDE DETECTION (Cross Product Math)
            # Determines if the bike is Left or Right of the slanted line
            val = (DIVIDER_P2[0] - DIVIDER_P1[0]) * (cy - DIVIDER_P1[1]) - \
                  (DIVIDER_P2[1] - DIVIDER_P1[1]) * (cx - DIVIDER_P1[0])
            
            # NOTE: If your 'Left' and 'Right' feel swapped, flip this '< 0' to '> 0'
            is_on_left = val > 0 

            # D. DIRECTION DETECTION
            direction_violation = False
            if track_id in track_history:
                prev_y = track_history[track_id]
                movement = cy - prev_y # (+) is DOWN, (-) is UP

                if is_on_left:
                    # LEFT LANE RULE: Must go UP. Violation if moving DOWN.
                    if movement > 3: 
                        direction_violation = True
                else:
                    # RIGHT LANE RULE: Must go DOWN. Violation if moving UP.
                    if movement < -3: 
                        direction_violation = True

            track_history[track_id] = cy

            # E. VISUAL LOGIC
            is_no_helmet = not helmet_detected
            
            # Expand box upward to include the Biker
            bike_h = y2 - y1
            rider_y1 = max(0, int(y1 - (bike_h * 0.6)))

            if is_no_helmet or direction_violation:
                color = (0, 0, 255) # RED for Violations
                violation_ids.add(track_id)
                
                # Construct Status Message
                errors = []
                if is_no_helmet: errors.append("NO_HELMET")
                if direction_violation: errors.append("WRONG_WAY")
                status = " & ".join(errors)

                # Save snapshot if not recently saved
                if (time.time() - last_saved_time) > COOLDOWN:
                    ts = datetime.now().strftime("%H%M%S")
                    cv2.imwrite(f"violations/ID{track_id}_{ts}.jpg", frame)
                    last_saved_time = time.time()
            else:
                color = (0, 255, 0) # GREEN for Safe
                status = "OK"

            # F. DRAW BOXES AND LABELS
            lane_name = "LEFT" if is_on_left else "RIGHT"
            cv2.rectangle(frame, (x1, rider_y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id} [{lane_name}] {status}", (x1, rider_y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # G. ON-SCREEN DASHBOARD
    cv2.rectangle(frame, (10, 10), (280, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"VIOLATIONS: {len(violation_ids)}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "STATUS: SYSTEM ACTIVE", (20, 68), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Display the result
    cv2.imshow("Enforcement AI v3.0", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Session Closed. Total Unique Violations Logged: {len(violation_ids)}")