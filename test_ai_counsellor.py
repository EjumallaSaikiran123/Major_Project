# test_ai_counsellor.py

import pytest
from flask import session
from app import create_app  # Import your app creation function

# Fixture to set up the Flask test client
@pytest.fixture
def client():
    app = create_app()  # Initialize Flask app with configurations
    app.config['TESTING'] = True  # Ensure the app is in testing mode
    with app.test_client() as client:
        yield client  # This allows the test to use the client to simulate requests

# Test case for Ai_Counsellor route
def test_ai_counsellor_integration(client):
    # Simulate session data before making the request
    with client.session_transaction() as sess:
        sess["name"] = "John Doe"
        sess["age"] = "22"
        sess["total_score"] = "150"
        sess["difficulty"] = "High"
        sess["suicide_status"] = "Low"
        sess["emotion_count"] = {"happiness": 5, "anger": 3, "sadness": 2}
        sess["highest_emotion"] = "Happiness"
        sess["user_text"] = "I feel stressed but hopeful."
        sess["emotion"] = "Happiness"
    
    # Simulate a POST request to the Ai_Counsellor route
    response = client.post("/Ai_Counsellor")
    
    # Assert that the status code is 200 (OK)
    assert response.status_code == 200
    
    # Check if the response contains the expected dynamic content
    assert b"Thank you for sharing your mental health assessment" in response.data
    assert b"Total Score: 150 / 250" in response.data
    assert b"Difficulty Level: High" in response.data
    assert b"Suicide Risk: Low" in response.data
    assert b"Emotion Count: {'happiness': 5, 'anger': 3, 'sadness': 2}" in response.data
    assert b"Dominant Emotion: Happiness" in response.data
    assert b"Student's Self-Reported Feeling: 'I feel stressed but hopeful.'" in response.data
    assert b"Student voice emotion detected is: 'Happiness'" in response.data
    
    # Check if the response contains therapeutic guidance
    assert b"Mental health remedies" in response.data
    assert b"Coping strategies" in response.data
    assert b"Professional help recommendations" in response.data

    # If suicide risk is high, check for urgent attention content
    with client.session_transaction() as sess:
        sess["suicide_status"] = "Yes"
    
    response_with_suicide = client.post("/Ai_Counsellor")
    
    # Assert that the response includes urgent suicide risk message
    assert b"Urgent Attention Needed: Suicide Risk Identified!" in response_with_suicide.data
    assert b"Access to a physical counselor for in-person support" in response_with_suicide.data

