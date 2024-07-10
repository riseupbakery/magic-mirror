#!/usr/bin/env python
import cv2
from insightface.app import FaceAnalysis
import numpy as np
import os
import math

def diagonal(x1, y1, x2, y2):
    x1, y1 = bbox[0], bbox[1]
    x2, y2 = bbox[2], bbox[3]
    diagonal_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return diagonal_length

def process_frame(frame):
    # Ensure the frame is in RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Ensure the frame is in uint8 format
    if rgb_frame.dtype != np.uint8:
        print(f"Converting frame dtype from {rgb_frame.dtype} to uint8")
        rgb_frame = rgb_frame.astype(np.uint8)
    
    #print(f"Processed frame with shape {rgb_frame.shape} and dtype {rgb_frame.dtype}")
    return rgb_frame

def encode_known_faces(known_faces_dir):
    known_faces = []
    known_names = []

    for name in os.listdir(known_faces_dir):
        if name.endswith(('.jpg', '.jpeg', '.png')):
            face_path = os.path.join(known_faces_dir, name)
            face_img = cv2.imread(face_path)
            face_encodings = app.get(face_img)
            if face_encodings:
                face_encoding = face_encodings[0].embedding
                known_faces.append(face_encoding)
                known_names.append(name.split('.')[0])
            else:
                print(f"No face detected in {name}")

    return known_faces, known_names

# Initialize the FaceAnalysis module
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# Encode known faces
known_faces_dir = 'known_faces'
known_faces, known_names = encode_known_faces(known_faces_dir)

print(f"Known faces: {known_names}")

# Capture video from the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture image")
        break
    
    try:
        # Process the frame to ensure correct format
        rgb_frame = process_frame(frame)
        # Perform face detection and recognition using InsightFace
        faces = app.get(rgb_frame)
        print(f"Found {len(faces)} face(s) in frame")
        
        # Draw bounding boxes around the detected faces and display names
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding

            # Compare face embedding with known faces
            distances = np.linalg.norm(known_faces - embedding, axis=1)
            min_distance_index = np.argmin(distances)

            if False:
                print(f"Distances: {distances}")
                print(f"Minimum distance index: {min_distance_index}")
                print(f"Minimum distance: {distances[min_distance_index]}")

            if distances[min_distance_index] < 20:  # Extremely low threshold for testing purposes
                name = known_names[min_distance_index]
            else:
                name = "Unknown"

            # Draw bounding box and display name
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            diagonal_length = diagonal(bbox[0], bbox[1], bbox[2], bbox[3])
            print(f"    Size: {int(diagonal_length):5d}   Who: {name:<7}")

    
    except Exception as e:
        print(f"Failed to process frame: {e}")
        continue
    
    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()