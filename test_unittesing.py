import pytest
from flask import session
from app import create_app  # your app factory
from unittest.mock import patch
import numpy as np
from app import predict_emotion, detect_face_emotion
# Fixture for test client
@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ---------------- TC08 - Check Response Includes Emotions ----------------
def test_response_contains_emotions(client):
    with client.session_transaction() as sess:
        sess.update({
            "name": "John",
            "age": "22",
            "total_score": "150",
            "difficulty": "High",
            "suicide_status": "Low",
            "emotion_count": {"happiness": 3},
            "highest_emotion": "Happiness",
            "user_text": "I feel okay.",
            "emotion": "Happiness"
        })
    response = client.post("/Ai_Counsellor")
    assert b"Dominant Emotion: Happiness" in response.data
    assert b"Student's Self-Reported Feeling: \"I feel okay.\"" in response.data

# ---------------- TC09 - Full Counselor Route ----------------
def test_ai_counsellor_full_route(client):
    with client.session_transaction() as sess:
        sess.update({
            "name": "Alice",
            "age": "20",
            "total_score": "180",
            "difficulty": "Moderate",
            "suicide_status": "Low",
            "emotion_count": {"anger": 2, "happiness": 4},
            "highest_emotion": "Happiness",
            "user_text": "I'm anxious but trying.",
            "emotion": "Happiness"
        })
    response = client.post("/Ai_Counsellor")
    assert response.status_code == 200
    assert b"Mental Health Analysis" in response.data
    assert b"Total Score: 180" in response.data

# ---------------- TC10 - Suicide Risk ----------------
def test_suicide_risk(client):
    with client.session_transaction() as sess:
        sess.update({
            "name": "Bob",
            "age": "21",
            "total_score": "200",
            "difficulty": "Severe",
            "suicide_status": "Yes",
            "emotion_count": {"sadness": 5},
            "highest_emotion": "Sadness",
            "user_text": "I feel hopeless.",
            "emotion": "Sadness"
        })
    response = client.post("/Ai_Counsellor")
    assert b"Urgent Attention Needed: Suicide Risk Identified!" in response.data


def test_predict_emotion():
    sample_audio = "C:/Users/ejuma_jrrzjzq/OneDrive/Desktop/Major Project Review 3/Datasets/TESS Toronto emotional speech set data/OAF_disgust/OAF_beg_disgust.wav"
    result = voicemodel.predict(sample_audio)
    assert result in ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISED"]

def test_detect_face_emotion():
    from keras.preprocessing.image import load_img, img_to_array
    image = load_img("C:/Users/ejuma_jrrzjzq/OneDrive/Desktop/Major Project Review 3/Datasets/Facial Emotion Images/test/fear/314.jpg", color_mode="grayscale", target_size=(48, 48))
    image = img_to_array(image)
    result = facemodel.predict(image)
    assert result in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
