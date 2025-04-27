# test_face_detection.py

import pytest
from app import create_app  # Import the create_app function

# Fixture to set up the Flask test client
@pytest.fixture
def client():
    # Create and configure the Flask app for testing
    app = create_app()
    
    # Set the app in testing mode
    app.config['TESTING'] = True
    
    # Use the Flask test client to simulate HTTP requests
    with app.test_client() as client:
        yield client  # This allows the test to use the client to send requests

# Test case for face detection route
def test_face_detection_integration(client):
    # Simulate a POST request to the '/face-detection' route
    response = client.post('/face-detection', data={'image': 'path_to_image'})
    
    # Assert that the status code is 200 (OK)
    assert response.status_code == 200
    
    # You can add more assertions here based on the response content
    assert b"Face Detection Successful" in response.data
