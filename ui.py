import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import random

st.markdown('<div class="main-container">', unsafe_allow_html=True)
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Multimedia Recommender")
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst))]
            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2) 
            # Save emotion only if it's not already captured
            if not emotion:
                np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1,circle_radius=1),connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Function to recommend movies by emotion
def recommend_movies_by_emotion(emotion):
    if emotion.lower() == "surprise":
        return "crime"
    elif emotion.lower() == "happy":
        return "action"
    elif emotion.lower() == "sad":
        return "horror"
    else:
        return "musical"

# Dropdown for recommendation options
option = st.selectbox("Select Recommendation", ["Recommendation on YouTube", "Recommendation on IMDb", "Generate Music"])

# Language input section
lang = ""
singer = ""
if option == "Recommendation on YouTube":
    lang = st.text_input("Language")
    singer = st.text_input("Singer")

# Button to capture emotion
if st.button("Capture Emotion") and (option == "Recommendation on YouTube" or option == "Recommendation on IMDb"):
    if lang and singer:
        webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)
        st.session_state["run"] = "emotion_capture"

if st.session_state["run"] == "emotion_capture":
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        if option == "Recommendation on YouTube" and lang and singer:
            webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        elif option == "Recommendation on IMDb" and lang:
            # Add IMDb recommendation logic here
            pass
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"

# Add a line separator
st.markdown("***")

# List of WAV files
wav_files = ['../l/bach_846.wav', '../l/bach_847.wav', '../l/bach_850.wav']

# Button to play random music
if option == "Generate Music" and st.button("Generate Music"):
    # Select a random WAV file
    random_wav = random.choice(wav_files)
    
    # Play the selected WAV file
    st.audio(random_wav, format='audio/wav')
    st.write("Playing piano automatically generated music:", random_wav)

st.markdown('</div>', unsafe_allow_html=True)
