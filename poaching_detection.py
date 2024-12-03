import cv2
import os
from ultralytics import YOLO
import pickle

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None, None, None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or unable to read more frames.")
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print("Error: No frames were read from the video.")
        return None, None, None

    print(f"Successfully read {len(frames)} frames from the video.")
    return frames, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def detect_poachers(model_path, video_path):
    # Load YOLO model
    model = YOLO(model_path)

    # Read video
    frames, width, height = read_video(video_path)
    if frames is None:
        return None, None  # Error reading video

    poacher_frames = []  # Frames with poachers detected

    for frame_idx, frame in enumerate(frames):
        # Preprocess frame
        frame_resized = cv2.resize(frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Run YOLO detection
        results = model.predict(frame_rgb, conf=0.1)
        detections = results[0].boxes.data.cpu().numpy()  # Ensure correct data format

        # Check for 'poacher' class (replace 0 with the class index for 'poacher')
        for detection in detections:
            cls = int(detection[5])  # Class ID
            if cls == 15:  # Replace 15 with your 'poacher' class index
                poacher_frames.append(frame_idx)
                break

    return poacher_frames, (width, height)

# Save model for Streamlit
model_path = 'E:\\python myfiles\\Poaching detection streamlit\\runs100\\detect\\train\\weights\\best.pt'
pickle.dump(model_path, open('model.pkl', 'wb'))
