import cv2
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace

# Function to detect faces using RetinaFace
def detect_faces_retina(frame):
    faces = RetinaFace.detect_faces(frame)
    face_regions = []
    if isinstance(faces, dict):
        for key in faces.keys():
            face = faces[key]["facial_area"]
            face_regions.append(face)
    return face_regions

# Function to get face embedding using DeepFace
def get_face_embedding(face_image):
    try:
        face_embedding = DeepFace.represent(face_image, model_name="Facenet", enforce_detection=False)
        return face_embedding[0]["embedding"]
    except Exception:
        return None

# Function to compare embeddings using cosine similarity
def cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

# Function to filter unique poachers based on face embeddings
def compare_faces_and_track_unique(poacher_frames, frames, similarity_threshold=0.7, max_no_face_frames=5):
    face_embeddings = []
    unique_poacher_frames = []
    no_face_frames = []  # Frames where no faces are detected

    for frame_idx in poacher_frames:
        frame = frames[frame_idx]

        # Detect faces using RetinaFace
        faces = detect_faces_retina(frame)

        # If no faces are detected, add the frame to `no_face_frames`
        if not faces:
            no_face_frames.append(frame_idx)
            continue

        # Process frames with detected faces
        for face in faces:
            x, y, w, h = face
            face_image = frame[y:h, x:w]

            # Extract face embedding
            embedding = get_face_embedding(face_image)
            if embedding is None:
                continue

            # Compare embedding with previously stored embeddings
            is_unique = True
            for prev_embedding in face_embeddings:
                similarity = cosine_similarity(embedding, prev_embedding)
                if similarity > similarity_threshold:
                    is_unique = False
                    break

            # If unique, add to the list
            if is_unique:
                unique_poacher_frames.append(frame_idx)
                face_embeddings.append(embedding)

    # Handle case when no faces are detected in all frames
    if len(unique_poacher_frames) == 0:
        unique_poacher_frames = no_face_frames[:max_no_face_frames]  # Take top N frames without faces

    return unique_poacher_frames

