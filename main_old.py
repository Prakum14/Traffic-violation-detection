import cv2
import os
from ultralytics import YOLO

# 1. SETUP: Define paths and folders
VIDEO_INPUT_PATH = "input_videos/sample_traffic.mp4" # Put your video file name here!
OUTPUT_BASE_DIR = "output_evidence"

# 2. INITIALIZE: Load the AI Model
# This downloads 'yolov8n.pt' (Nano) which is fast for laptops
print("Loading YOLOv8 Model...")
model = YOLO('models/yolov8n.pt') 

def process_traffic_video():
    # Open the video file
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_INPUT_PATH}")
        return

    frame_count = 0
    print("Processing started... Press 'q' to stop.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # End of video

        frame_count += 1
        
        # Only process every 5th frame to make it faster (Commercial Best Practice)
        if frame_count % 5 != 0:
            continue

        # 3. DETECTION: Run AI on the frame
        # We look for 'person' (ID: 0) and 'motorcycle' (ID: 3)
        results = model(frame, verbose=False)

        # 4. VIOLATION LOGIC: (Simplified for MVP)
        # For now, if we see a motorcycle, we will assume a violation 
        # just to test if the "Save to Folder" logic works.
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                
                if class_id == 3: # 3 is the ID for 'motorcycle'
                    print(f"Motorcycle detected in frame {frame_count}!")
                    
                    # 5. SAVE EVIDENCE: Snapshot of the violation
                    save_path = os.path.join(OUTPUT_BASE_DIR, "No_Helmet", f"violation_{frame_count}.jpg")
                    cv2.imwrite(save_path, frame)

        # 6. VISUALIZE: Show the video in a window
        annotated_frame = results[0].plot() # Draw boxes on the frame
        cv2.imshow("Traffic Monitor", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Processing Complete. Check your output_evidence folders.")

if __name__ == "__main__":
    process_traffic_video()