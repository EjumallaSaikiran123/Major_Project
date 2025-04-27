AI-Powered Personalized Therapy with Real-Time Emotion Detection

This project provides intelligent and personalized mental health support using real-time emotion detection powered by AI. It captures users' emotions through video/audio input, processes them using deep learning models, and delivers tailored therapeutic responses instantly.

Team Members:

    Ejumalla Saikiran

    G Manasa

    T Sai Vinay

Tech Stack:

    Backend: Python 3.x, Flask

    Machine Learning: TensorFlow / Keras, OpenCV, scikit-learn

    Emotion Detection: CNN, VGG16, LBP + SVM

    Frontend: HTML, CSS, JavaScript (optional Bootstrap)

    Others: MediaPipe/dlib for facial features, Flask APIs for module integration

Project Structure: /emotion-therapy-app

    modules/

        audio_emotion/ Audio-based emotion detection

        video_emotion/ Video/face-based emotion detection

        ai_response/ Personalized AI therapy suggestions

    static/ Static files (CSS, images, etc.)

    templates/ HTML templates (index.html, result.html)

    app.py Main Flask app (integrates all modules)

    requirements.txt Python dependencies

    README.md Project documentation

Development Strategy:

    Modular Design
    Each member implemented an independent module:

    Saikiran: Face/video-based emotion detection, AI Therapy Suggestion Implementation using mistralai from Hugging Face,Email & whatsapp communications implementation for Physical counsellors.

    Manasa: Audio-based emotion detection

    Vinay: User survey form

    Predefined Tech Stack
    A common tech stack (Flask, Python, ML libraries) was selected before development to avoid conflicts and ease integration.

    Independent Testing
    Each module was tested with sample data to ensure expected output and robustness.

    Integration via Flask APIs
    Modules were wrapped into Flask routes and called via API endpoints as per user input and interface flow.

    Error Handling & Logging
    All modules have try-except blocks and logging to handle exceptions gracefully and track execution.

    Code Documentation
    Inline comments and docstrings are provided in each function. This README file summarizes the project usage.

How to Run:

    Clone the Repository
    git clone https://github.com/yourusername/emotion-therapy-app.git
    cd emotion-therapy-app

    Set Up Virtual Environment (Optional)
    python -m venv venv
    source venv/bin/activate (On Windows: venv\Scripts\activate)

    Install Requirements
    pip install -r requirements.txt

    Run the Application
    python app.py

    Access in Browser
    Open http://127.0.0.1:5000 to interact with the app

Features:

    Detect emotions in real-time using facial expressions or voice

    Generate personalized therapy messages using AI

    Clean, modular design with scalable code structure

    Future-ready architecture for expansion into mobile apps or cloud APIs

Contact: For collaborations or questions:

    Ejumalla Saikiran* – ejumallasaikiran2708@gmail.com

    G Manasa – 22r15a0515@gcet.edu.in

    T Sai Vinay – 21r11a05f3@gcet.edu.in