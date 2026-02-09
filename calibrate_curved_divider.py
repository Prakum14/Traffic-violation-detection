import cv2
import numpy as np

"""
CURVED DIVIDER CALIBRATION TOOL
================================
This tool helps you define a CURVED road divider by clicking multiple points.

Instructions:
1. The video will pause on the first frame
2. Click points along the road divider from TOP to BOTTOM
   - Click as many points as needed to follow the curve
   - Minimum: 2 points (straight line)
   - Recommended: 4-6 points for smooth curves
3. The tool will draw a smooth curve through your points
4. Copy the coordinates to your main script

Controls:
- Left click: Add point to divider
- 'r': Reset all points
- 'u': Undo last point
- 'n': Next frame
- 'c': Complete and show final curve
- 'q': Quit
"""

# Global variables
points = []
frame = None
frame_copy = None

def interpolate_curve(points, num_interpolated=100):
    """Create smooth curve from points."""
    if len(points) < 2:
        return points
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, num_interpolated)
    
    x_new = np.interp(t_new, t, x_coords)
    y_new = np.interp(t_new, t, y_coords)
    
    return list(zip(x_new.astype(int), y_new.astype(int)))

def draw_divider(img, points):
    """Draw the divider on the image."""
    if len(points) == 0:
        return
    
    # Draw control points
    for i, point in enumerate(points):
        cv2.circle(img, point, 8, (0, 0, 255), -1)
        cv2.putText(img, f"{i+1}", (point[0] + 10, point[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw lines between points
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], (100, 100, 100), 2)
    
    # Draw smooth curve
    if len(points) >= 2:
        smooth_curve = interpolate_curve(points, num_interpolated=100)
        for i in range(len(smooth_curve) - 1):
            cv2.line(img, smooth_curve[i], smooth_curve[i + 1], (0, 255, 255), 3)
        
        # Draw side indicators
        if len(smooth_curve) > 50:
            mid_idx = len(smooth_curve) // 2
            mid_point = smooth_curve[mid_idx]
            
            # Calculate perpendicular direction
            prev_point = smooth_curve[max(0, mid_idx - 5)]
            next_point = smooth_curve[min(len(smooth_curve) - 1, mid_idx + 5)]
            
            dx = next_point[0] - prev_point[0]
            dy = next_point[1] - prev_point[1]
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                perp_dx = -dy / length * 80
                perp_dy = dx / length * 80
                
                # Left side
                left_point = (int(mid_point[0] + perp_dx), int(mid_point[1] + perp_dy))
                cv2.arrowedLine(img, mid_point, left_point, (255, 0, 0), 2)
                cv2.putText(img, "LEFT", left_point, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 0, 0), 2)
                
                # Right side
                right_point = (int(mid_point[0] - perp_dx), int(mid_point[1] - perp_dy))
                cv2.arrowedLine(img, mid_point, right_point, (0, 0, 255), 2)
                cv2.putText(img, "RIGHT", right_point, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)

def mouse_callback(event, x, y, flags, param):
    global points, frame, frame_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        
        # Redraw
        frame = frame_copy.copy()
        draw_divider(frame, points)
        
        # Add instruction
        cv2.putText(frame, f"Points: {len(points)} | Press 'c' when complete", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Curved Divider Calibration", frame)

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
    
    print("\n" + "="*70)
    print("CURVED DIVIDER CALIBRATION TOOL")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"Total Frames: {total_frames}")
    print("="*70)
    print("\nInstructions:")
    print("1. Click points along the road divider from TOP to BOTTOM")
    print("2. Add as many points as needed to follow the curve")
    print("3. Minimum 2 points, recommended 4-6 for smooth curves")
    print("4. The tool will draw a smooth yellow curve through your points")
    print("\nControls:")
    print("  Left Click - Add point")
    print("  'r' - Reset all points")
    print("  'u' - Undo last point")
    print("  'n' - Next frame")
    print("  'c' - Complete and show final result")
    print("  'q' - Quit")
    print("="*70)
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    frame_copy = frame.copy()
    
    cv2.namedWindow("Curved Divider Calibration")
    cv2.setMouseCallback("Curved Divider Calibration", mouse_callback)
    
    # Add instruction on frame
    cv2.putText(frame, "Click points along the divider (TOP to BOTTOM)", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Curved Divider Calibration", frame)
    
    current_frame = 0
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset
            points = []
            frame = frame_copy.copy()
            cv2.putText(frame, "Click points along the divider (TOP to BOTTOM)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Curved Divider Calibration", frame)
            print("Points reset. Click to mark divider curve.")
        elif key == ord('u'):
            # Undo last point
            if points:
                removed = points.pop()
                print(f"Removed point: {removed}")
                frame = frame_copy.copy()
                draw_divider(frame, points)
                cv2.putText(frame, f"Points: {len(points)} | Press 'c' when complete", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Curved Divider Calibration", frame)
        elif key == ord('n'):
            # Next frame
            ret, new_frame = cap.read()
            if ret:
                current_frame += 1
                frame = new_frame
                frame_copy = frame.copy()
                points = []
                cv2.putText(frame, "Click points along the divider (TOP to BOTTOM)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Curved Divider Calibration", frame)
                print(f"Frame {current_frame}/{total_frames}")
            else:
                print("End of video reached")
        elif key == ord('c'):
            # Complete
            if len(points) >= 2:
                print("\n" + "="*70)
                print("DIVIDER CURVE COMPLETE!")
                print("="*70)
                print(f"Total Points: {len(points)}")
                print("\nCopy this to your script:")
                print("="*70)
                print("DIVIDER_POINTS = [")
                for i, point in enumerate(points):
                    comma = "," if i < len(points) - 1 else ""
                    print(f"    {point}{comma}  # Point {i+1}")
                print("]")
                print("="*70)
                print("\nNote: RED arrow = LEFT side, BLUE arrow = RIGHT side")
                print("="*70)
                
                # Show final result
                result_frame = frame_copy.copy()
                draw_divider(result_frame, points)
                
                # Add big text
                cv2.putText(result_frame, "FINAL DIVIDER CURVE", 
                           (10, frame_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
                cv2.imshow("Curved Divider Calibration", result_frame)
                
                # Wait for user to review
                print("\nPress any key to close...")
                cv2.waitKey(0)
            else:
                print("Need at least 2 points! Click more points or press 'r' to reset.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(points) >= 2:
        print("\n" + "="*70)
        print("FINAL COORDINATES:")
        print("="*70)
        print("DIVIDER_POINTS = [")
        for i, point in enumerate(points):
            comma = "," if i < len(points) - 1 else ""
            print(f"    {point}{comma}")
        print("]")
        print("="*70)

if __name__ == "__main__":
    main()