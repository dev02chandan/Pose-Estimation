import streamlit as st
import cv2 
import numpy as np
import mediapipe as mp
from PIL import Image
import joblib
# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
#Loding pretrained model and scaler
loaded_model = joblib.load('./models/svc1.sav') 
loaded_scaler = joblib.load('./models/scaler.sav')  
# Function to extract landmarks and draw them on the image
def extract_landmarks(image):
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
    st.title("Pose Detection Using Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Extract landmarks
        landmarks, landmarked_image = extract_landmarks(image)

        if landmarks:
            features_scaled = loaded_scaler.transform([landmarks])
            prediction = loaded_model.predict(features_scaled)
            label = f'Predicted Category: {prediction[0]}'
            #cv2.putText(landmarked_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Resizing the landmarked image to a fixed size
            fixed_size = (400,200)  
            landmarked_image_resized = cv2.resize(landmarked_image, fixed_size, interpolation=cv2.INTER_AREA)
            if prediction[0]=='tree' or prediction[0]=='goddess':
                landmarked_image_resized = cv2.resize(landmarked_image, (400,410), interpolation=cv2.INTER_AREA)
                
            # Display the image with landmarks and label
            st.image(cv2.cvtColor(landmarked_image_resized, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.markdown(f"<h4 style='text-align: center; color: blue;'>{label}</h4>", unsafe_allow_html=True)
        else:
            st.write("No pose landmarks detected in the image.")


    st.title("Real-time Pose Estimation")
    st.info("Goto Sidebar and do accordingly")
    #st.info()
    
    # Use Webcam
    use_webcam = st.sidebar.button('Use Webcam', key='start_webcam')
    stop_webcam = st.sidebar.button('Stop Webcam', key='stop_webcam')
    if use_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        pose = mp_pose.Pose()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks, landmarked_image = extract_landmarks(frame)

            if landmarks:
                features_scaled = loaded_scaler.transform([landmarks])
                prediction = loaded_model.predict(features_scaled)
                cv2.putText(landmarked_image, f'Predicted Category: {prediction[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

           
            stframe.image(landmarked_image, channels="BGR")
           
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        cap.release()
        cv2.destroyAllWindows()
    
    
        

if __name__ == "__main__":
    main()
