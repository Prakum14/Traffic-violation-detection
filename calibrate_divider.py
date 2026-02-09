import cv2
import numpy as np

"""
DIVIDER CALIBRATION TOOL
========================
This tool helps you find the correct coordinates for your road divider.

Instructions:
1. The video will pause on the first frame
2. Click TWO points to define the divider line:
   - First click: Top point of the divider
   - Second click: Bottom point of the divider
3. The coordinates will be printed in the terminal
4. Copy these coordinates to your main script

Controls:
- Left click: Mark divider points
- 'r': Reset points
- 'n': Next frame
- 'q': Quit
"""

# Global variables
points = []
frame = None
frame_copy = None

def mouse_callback(event, x, y, flags, param):
    global points, frame, frame_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            
            # Draw the point
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(frame, f"P{len(points)}", (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # If we have 2 points, draw the line
            if len(points) == 2:
                cv2.line(frame, points[0], points[1], (0, 255, 255), 3)
                
                # Draw perpendicular indicators to show left/right sides
                mid_x = (points[0][0] + points[1][0]) // 2
                mid_y = (points[0][1] + points[1][1]) // 2
                
                # Calculate perpendicular direction
                dx = points[1][0] - points[0][0]
                dy = points[1][1] - points[0][1]
                length = np.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    # Perpendicular vector (rotated 90 degrees)
                    perp_dx = -dy / length * 50
                    perp_dy = dx / length * 50
                    
                    # Left side indicator
                    left_point = (int(mid_x + perp_dx), int(mid_y + perp_dy))
                    cv2.arrowedLine(frame, (mid_x, mid_y), left_point, (255, 0, 0), 2)
                    cv2.putText(frame, "LEFT", left_point, cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 0, 0), 2)
                    
                    # Right side indicator
                    right_point = (int(mid_x - perp_dx), int(mid_y - perp_dy))
                    cv2.arrowedLine(frame, (mid_x, mid_y), right_point, (0, 0, 255), 2)
                    cv2.putText(frame, "RIGHT", right_point, cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 0, 255), 2)
                
                print("\n" + "="*60)
                print("DIVIDER COORDINATES FOUND!")
                print("="*60)
                print(f"DIVIDER_P1 = {points[0]}")
                print(f"DIVIDER_P2 = {points[1]}")
                print("="*60)
                print("\nCopy these values to your main script!")
                print("Note: BLUE arrow shows RIGHT side, RED arrow shows LEFT side")
                print("="*60)
            
            cv2.imshow("Calibration Tool", frame)

def main():
    global points, frame, frame_copy
    
    video_path = input("Enter video path (or press Enter for 'input_videos/sample_traffic.mp4'): ").strip()
    if not video_path:
        video_path = "input_videos/sample_traffic.mp4"
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("\n" + "="*60)
    print("DIVIDER CALIBRATION TOOL")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"Total Frames: {total_frames}")
    print("="*60)
    print("\nInstructions:")
    print("1. Click the TOP point of the road divider")
    print("2. Click the BOTTOM point of the road divider")
    print("3. The line should separate the two lanes of traffic")
    print("\nControls:")
    print("  Left Click - Mark point")
    print("  'r' - Reset points")
    print("  'n' - Next frame")
    print("  'q' - Quit")
    print("="*60)
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    frame_copy = frame.copy()
    
    cv2.namedWindow("Calibration Tool")
    cv2.setMouseCallback("Calibration Tool", mouse_callback)
    
    cv2.imshow("Calibration Tool", frame)
    
    current_frame = 0
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset
            points = []
            frame = frame_copy.copy()
            cv2.imshow("Calibration Tool", frame)
            print("Points reset. Click again to mark divider.")
        elif key == ord('n'):
            # Next frame
            ret, new_frame = cap.read()
            if ret:
                current_frame += 1
                frame = new_frame
                frame_copy = frame.copy()
                points = []
                cv2.imshow("Calibration Tool", frame)
                print(f"Frame {current_frame}/{total_frames}")
            else:
                print("End of video reached")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(points) == 2:
        print("\n" + "="*60)
        print("FINAL COORDINATES:")
        print("="*60)
        print(f"DIVIDER_P1 = {points[0]}")
        print(f"DIVIDER_P2 = {points[1]}")
        print("="*60)

if __name__ == "__main__":
    main()
