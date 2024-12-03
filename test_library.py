import streamlit as st
from poaching_detection import detect_poachers, read_video
from face_detection import compare_faces_and_track_unique
import os
import cv2
import pickle

st.title("Poaching Detection System")

# Load model path
model = pickle.load(open('model.pkl', 'rb'))
#model_path = "path/to/best.pt"  # Replace this with actual model path
st.write("Loaded YOLO model:", model)

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Save video locally
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.write("Video uploaded successfully!")

    # Detect poachers
    st.write("Processing video...")
    try:
        poacher_frames, resolution = detect_poachers(model, video_path)
        print("***", poacher_frames)
        if poacher_frames is None:
            st.error("Could not process the video. Please try again.")
        elif len(poacher_frames) == 0:
            st.success("No poachers detected in the video!")
        else:
            st.warning(f"Poachers detected in {len(poacher_frames)} frames.")

            # Read video frames and display unique poacher frames
            frames, _, _ = read_video(video_path)
            unique_poacher_frames = compare_faces_and_track_unique(poacher_frames, frames)

            st.write(f"Displaying frames with unique poachers detected...")
            for frame_idx in unique_poacher_frames[:5]:  # Limit to 5 frames for display
                st.image(cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB), caption=f"Frame {frame_idx + 1}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Clean up temporary file
    os.remove(video_path)
