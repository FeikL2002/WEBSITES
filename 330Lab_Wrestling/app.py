import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import mediapipe as mp  # Standard import

# --- 330 LAB BRANDING ---
st.set_page_config(page_title="330 Lab AI Coach", layout="wide")
st.markdown("<h1 style='text-align: center; color: #32CD32;'>🧪 330 LAB: PERFORMANCE PORTAL</h1>", unsafe_allow_html=True)

# --- AI SETUP ---
# We use the standard solution path here
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@st.cache_resource
def load_pose():
    return mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5)

pose_tracker = load_pose()

# --- THE REST OF YOUR APP LOGIC ---
athlete_name = st.sidebar.text_input("Wrestler Name:", "Wrestler")
run_cam = st.checkbox("ACTIVATE CAMERA")
FRAME_WINDOW = st.image([])

if run_cam:
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and run_cam:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_tracker.process(rgb_frame)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()