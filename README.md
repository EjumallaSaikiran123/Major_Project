AI-Powered Personalized Therapy with Real-Time Emotion Detection

This project provides intelligent and personalized mental health support using real-time emotion detection powered by AI. It captures users' emotions through video/audio input, processes them using deep learning models, and delivers tailored therapeutic responses instantly.


Abstract:

In the current academic scenario, the student mental health has emerged as a pressure anxiety, with increasing examples of stress, anxiety, and emotional crisis, the consequences of learning and adverse effects on individual development. Traditional counselling approaches often fail to catch early indicators of psychological imbalance, especially in introverted individuals who cannot openly communicate their emotional conflicts. This research proposes a multimodal emotion -ware system to quickly detect mental health issues in students using affectionate computing techniques. The proposed system integrates facial emotion recognition, voice emotion analysis and psychological survey evaluation to assess the overall, non-guspath of a student's emotional status. The methodology involves detecting real-time facial expression, using a tone and pitch analysing the LSTM-RNN model using the Convolution Neural Network (CNNs), voice emotion classification, and the mental health suggestions and mental health tips and mental health suggestions and mental health suggestions and remedies are provided by Mistral AI (LLM). This multimodal data fusion enhances the ability of the system to detect emotional instability and behavioural patterns that may indicate and treat stress, depression, or potentially suicidal ideas. In this study, emotion detection was performed using two primary modalities—Facial Emotion Detection and Voice Emotion Detection—each evaluated individually and in combination to assess the effectiveness of multi-modal fusion. The Facial Emotion Detection model achieved an accuracy of 63% using CNN, indicating moderate performance due to challenges like occlusion, lighting variations, and subtle facial cues. In contrast, the Voice Emotion Detection model demonstrated a significantly higher accuracy of 93% using LSTM, owing to the richness of emotional expression in speech features such as pitch, tone, and tempo. To enhance the overall robustness of the system and provide deeper insights into emotional states, a collaborative multi-modal approach was employed. While a weighted SoftMax fusion technique was explored—assigning 70% weight to the voice model and 30% to the facial model for combined emotion prediction—the core objective was not merely to merge outputs into a single prediction. Instead, both the voice and facial emotion recognition models operate in parallel, providing complementary perspectives on the user's emotional state. This dual-channel analysis enables the system to capture nuanced variations in emotional expression that may be missed by unimodal approaches. With this cooperative strategy, the system achieved an enhanced understanding of user emotions, leading to improved mental health support. The approach demonstrates that integrating multiple modalities, even without strict fusion, can significantly aid in the identification and analysis of complex emotional patterns, reinforcing the value of multi-modal systems in real-world therapeutic applications.. The conclusion underlines the system's ability to promote an emotionally intelligent educational environment that proactively support student well-being. Future enhancements include expanding the dataset diversity, incorporating physiological signals like heart rate and EEG, and integrating the system with institutional learning management platforms to offer personalized mental health dashboards for students and counsellors. This research marks a significant step toward building empathetic, AI-driven solutions for mental health awareness and intervention in academic settings.
Keywords
Human Computer Interaction, Convolution Neural Network, Recurrent Neural Network, Long Short-Term Memory, Survey Form (Difficulty Scoring System of Mental Status)


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
