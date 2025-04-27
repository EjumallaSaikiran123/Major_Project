import os
import io
import pytest
from app import app
from werkzeug.datastructures import FileStorage
from unittest.mock import patch

@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client

def test_process_route_with_valid_data(client):
    # Dummy form data
    data = {
        "name": "John Doe",
        "age": "23",
        "roll": "12345",
        "year": "4",
        "branch": "CSE",
        "section": "A",
        "phone_number": "9876543210",
        "user_thoughts": "Feeling anxious lately."
    }

    # Add mental health values as required
    for field in [
        "mental_state", "talk_feelings", "overwhelmed", "Habitchange", "Happiness",
        "study", "academic_pressure", "motivation", "anxious", "course", "sad",
        "lonely", "conflicts", "social", "relationships", "energylevels", "sleep",
        "trust", "professional", "future_prospects", "challenges", "mechanisms",
        "view", "satisfaction", "suicidal-thoughts"
    ]:
        data[field] = "3"

    # Load test audio (make sure it's small for test performance)
    test_audio_path = "C:/Users/ejuma_jrrzjzq/OneDrive/Desktop/Major Project Review 3/Datasets/TESS Toronto emotional speech set data/OAF_disgust/OAF_beg_disgust.wav"
    with open(test_audio_path, "rb") as f:
        audio_data = (io.BytesIO(f.read()), "test_audio.wav")

    # Patch the prediction to skip actual model prediction (mock voice emotion)
    with patch("app.predict_emotion", return_value="HAPPY"):
        response = client.post(
            "/process",
            data={**data, "audio": audio_data},
            content_type="multipart/form-data",
            follow_redirects=True
        )

    # Assertions
    assert response.status_code == 200
    assert b"John Doe" in response.data
    assert b"HAPPY" in response.data  # From patched emotion
    assert b"Feeling anxious lately" in response.data
