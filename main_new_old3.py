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

# DEFINE THE WHITE STRIP
DIVIDER_P1 = (1200, 20) 
DIVIDER_P2 = (-10, 800)

os.makedirs('violations', exist_ok=True)
track_history = {} 
violation_ids = set() 
last_saved_time = 0
COOLDOWN = 2 

cap = cv2.VideoCapture("input_videos/sample_traffic.mp4")

print("System Initialized. Monitoring Dual-Lane Traffic...")

#Keep everything fixed above this line. Only MODEL_ID can be changed.
##################################################################################################################################

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # --- VISUAL REFERENCE: DRAW THE DIVIDER ---
    cv2.line(frame, DIVIDER_P1, DIVIDER_P2, (0, 255, 255), 3) # Bold Yellow Line
    cv2.putText(frame, "ROAD DIVIDER", (DIVIDER_P1[0]-150, DIVIDER_P1[1]+30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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

            # C. DIRECTION LOGIC
            # Cross product to determine side of divider
            val = (DIVIDER_P2[0] - DIVIDER_P1[0]) * (cy - DIVIDER_P1[1]) - \
                  (DIVIDER_P2[1] - DIVIDER_P1[1]) * (cx - DIVIDER_P1[0])
            
            is_on_left = val > 0 # Matches your previous calibrated logic

            direction_violation = False
            if track_id in track_history:
                movement = cy - track_history[track_id]
                # Left side must go UP (negative movement), Right side must go DOWN (positive)
                if is_on_left and movement > 3: 
                    direction_violation = True
                elif not is_on_left and movement < -3: 
                    direction_violation = True

            track_history[track_id] = cy

            # D. HELMET LOGIC (Wide Search)
            rider_has_helmet = False
            for p in api_preds:
                px, py = int(p['x']), int(p['y'])
                label = p['class'].lower()
                conf = p['confidence']

                # Lowered confidence threshold to 0.25 to reduce false "No_helmet" flags
                if ("helmet" in label or "safety" in label) and conf > 0.25:
                    # Look for helmet within a wide horizontal margin of the bike
                    if (x1 - 40) <= px <= (x2 + 40):
                        # Ensure the helmet is roughly at the top half of the bike or above
                        if py < y2 - ( (y2-y1) * 0.3 ): 
                            rider_has_helmet = True
                            break
            
            # E. PREPARE LABELS
            is_no_helmet = not rider_has_helmet
            errors = []
            
            if is_no_helmet: errors.append("No_helmet")
            if direction_violation: errors.append("Wrong_direction")

            bike_h = y2 - y1
            rider_y_top = max(0, int(y1 - (bike_h * 0.6)))

            if not errors:
                status = "OK"
                color = (0, 255, 0) # Green
            else:
                status = " & ".join(errors)
                color = (0, 0, 255) # Red
                violation_ids.add(track_id)
                
                # Snapshot saving logic
                if (time.time() - last_saved_time) > COOLDOWN:
                    cv2.imwrite(f"violations/ID{track_id}_{int(time.time())}.jpg", frame)
                    last_saved_time = time.time()

            # F. DRAW BOX AND CLEAN LABEL
            cv2.rectangle(frame, (x1, rider_y_top), (x2, y2), color, 2)
            cv2.putText(frame, status, (x1, rider_y_top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # G. DASHBOARD
    cv2.rectangle(frame, (10, 10), (220, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Violations: {len(violation_ids)}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Traffic AI Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()