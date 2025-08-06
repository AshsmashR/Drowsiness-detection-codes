import cv2
import time
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Blink detection variables
blink_count = 0
eye_closed_frames = 0
eye_closed_threshold = 3  # Frames threshold for blink detection

blink_duration = 0
blink_start_time = 0
blink_durations = []  # Store all blink durations
blink_timestamps = []  # Store timestamps of blinks
fps = 60  # Assume 30 fps, will be calculated dynamically

def calculate_blink_duration(closed_frames, fps):
    """Calculate blink duration in milliseconds"""
    return (closed_frames / fps) * 1000

def calculate_blink_rate(blink_timestamps, window_seconds=60):
    """Calculate blinks per minute over a time window"""
    if len(blink_timestamps) < 2:
        return 0
    
    current_time = time.time()
    recent_blinks = [t for t in blink_timestamps if current_time - t <= window_seconds]
    
    if len(recent_blinks) == 0:
        return 0
    
    # Calculate blinks per minute
    time_span = min(window_seconds, current_time - recent_blinks[0])
    if time_span > 0:
        return (len(recent_blinks) / time_span) * 60
    return 0



def calculate_perclose_ratio(eye_landmarks=None):
    """
    PERCLOSE (Percentage of Eye Closure) calculation
    For now using frame-based approximation since we don't have eye landmarks
    This is a simplified version - ideally would use eye aspect ratio
    """
    # This is a placeholder - in a real implementation you'd use:
    # - Eye aspect ratio (EAR) from facial landmarks
    # - Distance between upper and lower eyelid
    # For now, we'll use closed_frames as a proxy
    return eye_closed_frames

# Check if 
# 
# camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Calculate actual FPS
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 60  # Default fallback

print(f"Blink detection started. FPS: {fps}")
print("Press ESC to quit.")

# Timing variables
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame")
        break

    frame_count += 1
    current_time = time.time()

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    eyes_detected = False
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Define regions of interest
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(15, 15))
        
        # Check if eyes are detected
        if len(eyes) >= 1:
            eyes_detected = True
            # Draw rectangles around detected eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    # Blink detection logic with duration tracking
    if not eyes_detected:
        if eye_closed_frames == 0:
            blink_start_time = current_time
        eye_closed_frames += 1
        
        # Calculate current PERCLOSE ratio
        perclose_ratio = calculate_perclose_ratio()
        
    else:
        # If eyes were closed for sufficient frames, count as a blink
        if eye_closed_frames >= eye_closed_threshold:
            blink_count += 1
            blink_end_time = current_time
            
            # Calculate blink duration
            blink_duration = calculate_blink_duration(eye_closed_frames, fps)
            blink_durations.append(blink_duration)
            blink_timestamps.append(blink_end_time)
            
            print(f"Blink #{blink_count} detected! Duration: {blink_duration:.1f}ms")
            
        eye_closed_frames = 0
    
    blink_rate = calculate_blink_rate(blink_timestamps, window_seconds=60)
    avg_blink_duration = np.mean(blink_durations) if blink_durations else 0
    
    # Display information on frame
    cv2.putText(frame, f"Blinks: {blink_count}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Rate: {blink_rate:.1f} bpm", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Last Duration: {blink_duration:.1f}ms", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Avg Duration: {avg_blink_duration:.1f}ms", (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"PERCLOSE: {eye_closed_frames}", (20, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Show the frame
    cv2.imshow('Blink Detection - OpenCV', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Final statistics
total_time = time.time() - start_time
print(f"\n=== BLINK DETECTION SUMMARY ===")
print(f"Total runtime: {total_time:.1f} seconds")
print(f"Total blinks: {blink_count}")
print(f"Average blink rate: {(blink_count/total_time)*60:.1f} blinks per minute")
if blink_durations:
    print(f"Average blink duration: {np.mean(blink_durations):.1f}ms")
    print(f"Min blink duration: {min(blink_durations):.1f}ms")
    print(f"Max blink duration: {max(blink_durations):.1f}ms")
    print(f"Blink durations: {[f'{d:.1f}ms' for d in blink_durations]}")
else:
    print("No blinks detected during session")
    
