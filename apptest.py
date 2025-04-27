# Import necessary modules from Flask
from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify  
# Flask: Core framework for the web application.
# render_template: Renders HTML templates.
# request: Handles incoming HTTP requests.
# redirect, url_for: Used for URL redirection.
# session: Manages user sessions.
# Response: Allows creating HTTP responses.
# jsonify: Converts Python data to JSON format for API responses.

import sqlite3  
# SQLite3: Lightweight database to store user data and other application-related information.

import cv2  
# OpenCV: Used for image processing, facial recognition, and computer vision-related tasks.

import numpy as np  
# NumPy: Provides support for numerical operations, especially for handling arrays and matrices.

import tensorflow as tf  
from tensorflow import keras  
from tensorflow.keras.models import load_model  
# TensorFlow & Keras: Used for deep learning and machine learning model implementation.
# load_model: Loads pre-trained models for facial recognition, emotion detection, etc.

from flask_bcrypt import Bcrypt  
# Flask-Bcrypt: Provides hashing functions for securely storing user passwords.

from flask_session import Session  
# Flask-Session: Manages user sessions (storing session data in a database or filesystem).

from collections import Counter  
# Counter: A data structure from the collections module to count occurrences of elements in a list.

import os  
# OS Module: Provides functions to interact with the operating system (e.g., file handling).

import joblib  
# Joblib: Used for saving and loading machine learning models efficiently.

import librosa  
# Librosa: A Python library for audio analysis, useful for speech recognition and emotion detection.

from werkzeug.utils import secure_filename  
# secure_filename: Ensures uploaded file names are safe for storage (prevents security vulnerabilities).

from langchain.llms import HuggingFaceEndpoint  
# HuggingFaceEndpoint: Enables integration with Hugging Face models for NLP tasks.

from langchain import PromptTemplate  
# PromptTemplate: Helps in structuring prompts when interacting with language models.

from dotenv import load_dotenv  
# load_dotenv: Loads environment variables from a .env file, useful for securing API keys and configurations.

import pickle

from twilio.rest import Client


import pdfkit
import smtplib
from email.message import EmailMessage


app = Flask(__name__)




# ------------------ CONFIG ------------------
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------ VOICE EMOTION DETECTION ------------------
voicemodel = load_model("models/emotion_model.h5")
label_encoder = joblib.load("models/label_encoder.pkl")

emotion_mapping = {
    'YAF_angry': 'ANGRY', 'YAF_disgust': 'DISGUST', 'YAF_fear': 'FEAR',
    'YAF_happy': 'HAPPY', 'YAF_neutral': 'NEUTRAL', 'YAF_sad': 'SAD',
    'YAF_pleasant_surprised': 'SURPRISED',
    'OAF_angry': 'ANGRY', 'OAF_disgust': 'DISGUST', 'OAF_Fear': 'FEAR',
    'OAF_happy': 'HAPPY', 'OAF_neutral': 'NEUTRAL', 'OAF_Sad': 'SAD',
    'OAF_Pleasant_surprise': 'SURPRISED'
}

def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    features = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    return features

def detect_voice_emotion(audio_file):
    features = extract_features(audio_file)
    features = features[np.newaxis, np.newaxis, :]  # Shape for CNN
    prediction = voicemodel.predict(features)
    predicted_label = np.argmax(prediction)
    raw_emotion = label_encoder.inverse_transform([predicted_label])[0]
    return emotion_mapping.get(raw_emotion, "NEUTRAL")


# ------------------ FACE EMOTION DETECTION ------------------
facemodel = load_model("models/emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
emotion_count = Counter()

def extract_face_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_face_emotion(image):
    img = extract_face_features(image)
    prediction = facemodel.predict(img)
    prediction_label = labels[prediction.argmax()]
    return prediction_label

def generate_frames():
    webcam = cv2.VideoCapture(0)
    while True:
        success, im = webcam.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (p, q, r, s) in faces:
                image = gray[q:q + s, p:p + r]
                cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                prediction_label = detect_face_emotion(image)
                emotion_count[prediction_label] += 1
                cv2.putText(im, '%s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            ret, buffer = cv2.imencode('.jpg', im)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

