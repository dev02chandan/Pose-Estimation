import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to extract landmarks and draw them on the image
def extract_landmarks(image, pose):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        row = []
        for landmark in landmarks:
            row.extend([landmark.x, landmark.y, landmark.z])
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return row, image
    else:
        return None, image

def main():
    st.title("Real-time Pose Estimation")
    
    # Use Webcam
    use_webcam = st.sidebar.button('Use Webcam')
    if use_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        pose = mp_pose.Pose()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks, landmarked_image = extract_landmarks(frame, pose)

            stframe.image(landmarked_image, channels="BGR")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
