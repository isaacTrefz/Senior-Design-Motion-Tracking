import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import sys
import csv  # Added for CSV export

# 1. SETUP: Configuration for Multi-Person
# base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO, # optimized for video consistency
    num_poses=2,  # <--- SET THIS TO 2 (or more)
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.8,
    output_segmentation_masks=False
)

# Initialize the Landmarker
with vision.PoseLandmarker.create_from_options(options) as landmarker:

    # Use argument for video file, or default to webcam if none provided
    video_path = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(video_path)

    # --- NEW: SETUP VIDEO WRITER ---
    # Get video properties for the writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Default to 30 if webcam doesn't report it
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_video = cv2.VideoWriter('output_tracked.mp4', fourcc, fps, (width, height))

    # --- NEW: SETUP CSV WRITER ---
    csv_file = open('pose_data.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # Create the CSV Header
    # Structure: Timestamp, Person_ID, then x,y,z,vis,pres for all 33 landmarks
    csv_header = ['Timestamp_ms', 'Person_ID']
    for i in range(33):
        csv_header.extend([f'x_{i}', f'y_{i}', f'z_{i}', f'vis_{i}', f'pres_{i}'])
    csv_writer.writerow(csv_header)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. CONVERT: MediaPipe needs mp.Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Calculate timestamp (required for VIDEO mode)
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # 4. DETECT: Run the multi-person detection
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # 5. SETUP DRAWING:
        # We need to create the image copy to draw on
        annotated_image = frame.copy()
        
        # Get the list of landmarks
        pose_landmarks_list = detection_result.pose_landmarks

        # Define distinct colors (BGR format): Green for Index 0, Blue for Index 1, Red for Index 2
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)] 

        # Only loop if landmarks were actually found
        if pose_landmarks_list:
            # Loop through each person found
            for idx, pose_landmarks in enumerate(pose_landmarks_list):
                
                # --- NEW: WRITE TO CSV ---
                # Create a row for this person in this frame
                csv_row = [frame_timestamp_ms, idx]
                for landmark in pose_landmarks:
                    # Append all data points for this specific landmark
                    csv_row.extend([
                        landmark.x, 
                        landmark.y, 
                        landmark.z, 
                        landmark.visibility, 
                        landmark.presence
                    ])
                csv_writer.writerow(csv_row)

                # --- DRAWING LOGIC ---
                # Select color based on index (modulo to prevent crash if > 3 people)
                color = colors[idx % len(colors)]
                
                # Draw standard skeleton first
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                    for landmark in pose_landmarks
                ])
                
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
                
                # DRAW AN ID TEXT ON THEIR CHEST
                # Landmark 11 (left shoulder) and 12 (right shoulder) are good anchors
                if len(pose_landmarks) > 12:
                    # Get pixel coordinates for the shoulder
                    x_px = int(pose_landmarks[11].x * frame.shape[1])
                    y_px = int(pose_landmarks[11].y * frame.shape[0]) - 20
                    
                    # Draw the ID number
                    cv2.putText(annotated_image, f"ID: {idx}", (x_px, y_px), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show the live feed
        cv2.imshow('Multi-Person Feed', annotated_image)
        
        # --- NEW: WRITE FRAME TO VIDEO FILE ---
        out_video.write(annotated_image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
out_video.release() # Release the video writer
csv_file.close()    # Close the CSV file
cv2.destroyAllWindows()
