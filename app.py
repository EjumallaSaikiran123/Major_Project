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
bcrypt = Bcrypt(app)

# Configure session
app.secret_key = "supersecretkey"
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# Initialize database
def connect_db():
    return sqlite3.connect('users.db')

def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

create_table()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        conn = connect_db()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            return redirect(url_for('login'))  
        except sqlite3.IntegrityError:
            return "Username already exists!"
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and bcrypt.check_password_hash(user[0], password):
            session['user'] = username  
            session['detected_emotions'] = []  
            return redirect(url_for('dashboard'))  
        else:
            return "Invalid username or password"

    return render_template('login.html')


def create_app():
    create_table()
    return app

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html', username=session['user'])
    return redirect(url_for('login'))


# ---------------------------------------------------------------------------------------------------------------------------------------------------#


UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model and label encoder
voicemodel = load_model("models/emotion_model.h5")
label_encoder = joblib.load("models/label_encoder.pkl")

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Emotion mapping (to map raw labels to clean ones)
emotion_mapping = {
    'YAF_angry': 'ANGRY', 'YAF_disgust': 'DISGUST', 'YAF_fear': 'FEAR',
    'YAF_happy': 'HAPPY', 'YAF_neutral': 'NEUTRAL', 'YAF_sad': 'SAD',
    'YAF_pleasant_surprised': 'SURPRISED',
    'OAF_angry': 'ANGRY', 'OAF_disgust': 'DISGUST', 'OAF_Fear': 'FEAR',
    'OAF_happy': 'HAPPY', 'OAF_neutral': 'NEUTRAL', 'OAF_Sad': 'SAD',
    'OAF_Pleasant_surprise': 'SURPRISED'
}

# Extract features from the audio file
def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    features = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    return features

# Predict emotion from the audio file
def predict_emotion(audio_file):
    features = extract_features(audio_file)
    features = features[np.newaxis, np.newaxis, :]  # Shape for CNN
    prediction = voicemodel.predict(features)
    predicted_label = np.argmax(prediction)
    raw_emotion = label_encoder.inverse_transform([predicted_label])[0]
    return emotion_mapping.get(raw_emotion, "NEUTRAL")


# ---------------------------------------------------------------------------------------------------------------------------------------------------#



#Emotion Detector Web

# Load the trained model
facemodel = load_model("models/emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Dictionary to store emotion frequency
emotion_count = Counter()

def extract_face_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

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
                img = extract_face_features(image)
                pred = facemodel.predict(img)
                prediction_label = labels[pred.argmax()]
                emotion_count[prediction_label] += 1
                cv2.putText(im, '%s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            ret, buffer = cv2.imencode('.jpg', im)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------------------------------------------------------------------------------------------------#




@app.route("/process", methods=["POST"])
def process():
    if request.method == "POST":
        # Collect form values
        name = request.form.get("name")
        age = request.form.get("age")
        roll = request.form.get("roll")
        year = request.form.get("year")
        branch = request.form.get("branch")
        section = request.form.get("section")
        pn = request.form.get("phone_number")

        # List of input field names
        fields = [
            "mental_state", "talk_feelings", "overwhelmed", "Habitchange", "Happiness",
            "study", "academic_pressure", "motivation", "anxious", "course", "sad",
            "lonely", "conflicts", "social", "relationships", "energylevels", "sleep",
            "trust", "professional", "future_prospects", "challenges", "mechanisms",
            "view", "satisfaction", "suicidal-thoughts"
        ]

        # Convert form values to integers (default to 0 if missing)
        values = [int(request.form.get(field, 0)) for field in fields]

        # Calculate the sum of the values
        total_score = sum(values)

        # Determine difficulty level
        if 1 <= total_score <= 25:
            difficulty = "Normal"
        elif 26 <= total_score <= 50:
            difficulty = "Moderate"
        elif 51 <= total_score <= 75:
            difficulty = "High"
        elif 76 <= total_score <= 100:
            difficulty = "Very High"
        elif 101 <= total_score <= 125:
            difficulty = "Extreme"
        else:
            difficulty = "Undefined"

        
        # Check for suicidal status
        suicidal_thoughts = int(request.form.get("suicidal-thoughts", 0))
        suicide_status = "Yes" if suicidal_thoughts in [4, 5] else "No"
        
        # Emotion detection results
        highest_emotion = max(emotion_count, key=emotion_count.get, default="No Data")
        
        if 'audio' not in request.files:
           return "No audio file provided", 400

        file = request.files['audio']
        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        emotion = predict_emotion(filepath)


        user_text = request.form.get("user_thoughts")
        session["name"] = name
        session["age"] = age
        session["roll"]=roll
        session["year"]=year
        session["branch"]=branch
        session["section"]=section
        session["pn"]=pn
        session["total_score"] = total_score
        session["difficulty"] = difficulty
        session["suicide_status"] = suicide_status
        session["emotion_count"] = dict(emotion_count)
        session["highest_emotion"] = highest_emotion
        session["emotion"] = emotion
        session["user_text"] = user_text

        return render_template(
            "report.html",
            name=name,
            age=age,
            roll=roll,
            year=year,
            branch=branch,
            section=section,
            pn=pn,
            total_score=total_score,
            difficulty=difficulty,
            suicide_status=suicide_status,
            emotion_count=dict(emotion_count),
            highest_emotion=highest_emotion,
            user_text=user_text,
            emotion=emotion
        )


# ---------------------------------------------------------------------------------------------------------------------------------------------------#



def create_app():
    app = Flask(__name__)
    
    # Configurations for testing, or any other configurations you need
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'your_secret_key'
    
    # Initialize your routes
    @app.route('/face-detection', methods=['POST'])
    def face_detection():
        # Your face detection logic goes here (for simplicity, I'm using a placeholder response)
        return "Face Detection Successful", 200
    
    return app

#---------------------------------------------------------------------------------------------------------------------------------#
@app.route("/Ai_Counsellor", methods=["POST"])
def AiCounsellor():
    # Retrieve stored session data
    load_dotenv()
    
    # Get the secret key

    name = session.get("name", "Anonymous")
    age = session.get("age", "Unknown")
    total_score = session.get("total_score", "0")
    difficulty = session.get("difficulty", "Normal")
    suicide_status = session.get("suicide_status", "Low")
    emotion_count = session.get("emotion_count", {})
    highest_emotion = session.get("highest_emotion", "Neutral")
    user_text = session.get("user_text","Not entered")
    emotion = session.get("emotion")

    # Generate dynamic question for AI
    question = f"""
    Hello {name},  

    Thank you for sharing your mental health assessment. Based on the evaluation, here is a personalized analysis of your current well-being:mention like this and assume you ar a therapist or counsellor.  
    A student named {name} (Age: {age}) has undergone a mental health assessment. Below are the findings:

    **📝 Mental Health Analysis:**
    - **Total Score:** {total_score} / 250 → Reflects the intensity of distress from the survey.
    - **Difficulty Level:** {difficulty} → Assessed from the responses.
    - **Suicide Risk:** {suicide_status}
    - **Emotion Count:** {emotion_count} → Recognized emotions from facial analysis.
    - **Dominant Emotion:** {highest_emotion} → The most detected emotion.
    - **Student's Self-Reported Feeling:** "{user_text}"
    - **Student voice emotion detected is:** "{emotion}"

    🔹 The **Total Score (out of 250)** and the **difficulty level** indicate the mental challenges the student is facing.  
    🔹 The **Emotion Count and Dominant Emotion** are obtained through the emotion detection system.  
    🔹 The **Student Feeling** is based on the user's self-reported input.  

    📌 **Therapeutic Guidance Required:**  
    Based on these insights, provide the student with:  
    - **💡 Mental health remedies**  
    - **🛠 Coping strategies**  
    - **👨‍⚕️ Professional help recommendations**  
    Complete it in 500 words.
    """

    if suicide_status.lower() == "yes":
        question += """
    🚨 **Urgent Attention Needed: Suicide Risk Identified!**  
     - This institution provides access to a **physical counselor** for in-person support.  
     - Please provide **immediate coping strategies**, **positive affirmations**, and **professional recommendations** to assist students experiencing suicidal thoughts.  
        """

    # Define prompt template
    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Format prompt
    formatted_prompt = prompt.format(question=question)
    
    llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",  # Model Repo
    temperature=0.7,  # Set explicitly
    max_length=500,  # Set explicitly
    token="HUGGINGFACEHUB_API_TOKEN"  # Replace with your actual Hugging Face API Token
    )
    # Get AI response
    generated_response = llm(formatted_prompt)
    session["generated_response"] = generated_response

    return render_template("suggestion.html", response=generated_response)

# ---------------------------------------------------------------------------------------------------------------------------------------------------#


PDF_FOLDER = "static/userpdf"
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

pdfkit_config = pdfkit.configuration(wkhtmltopdf=os.getenv("WKHTMLTOPDF_PATH", r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'))




# ---------------------------------------------------------------------------------------------------------------------------------------------------#


# Load environment variables
load_dotenv()

# Email Configuration
EMAIL_ADDRESS = "aicounsellorgcet@gmail.com"
EMAIL_PASSWORD = "EMAIL_PASS"  # Use environment variables instead of hardcoding
recipient_email = "22r15a0514@gcet.edu.in"  # Default recipient if session email is missing

def is_valid_email(recipient_email):
    """Validate email format."""
    import re
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, recipient_email) is not None

# ---------------------------------------------------------------------------------------------------------------------------------------------------#



def send_email_with_attachment_to_gcetcounsellor(recipient_email, pdf_path, pdf_filename):
    """Send an email with the generated PDF attachment."""
    if not recipient_email:
        print("Error: No recipient email provided!")
        return
    
    if not is_valid_email(recipient_email):
        print(f"Error: Invalid email format -> {recipient_email}")
        return

    msg = EmailMessage()
    
    name = session.get('name', 'Unknown')
    
    msg["Subject"] = (f"{name}'s Mental Health Report")
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = recipient_email
    msg.set_content("Please find your mental health report attached.")

    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
        msg.add_attachment(pdf_data, maintype="application", subtype="pdf", filename=pdf_filename)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"Email sent successfully to {recipient_email}")
    except Exception as e:
        print(f"Error sending email: {e}")

 # ---------------------------------------------------------------------------------------------------------------------------------------------------#


def send_email_with_attachment_to_gcetStudent(user_email, pdf_path, pdf_filename):
    """Send an email with the generated PDF attachment."""
    if not user_email:
        print("Error: No recipient email provided!")
        return
    
    if not is_valid_email(user_email):
        print(f"Error: Invalid email format -> {user_email}")
        return

    msg = EmailMessage()
    user_email = session.get('user', 'Unknown')

    name = session.get('name', 'Unknown')
    generated_response=session.get('generated_response','Not gererated')
    suicide_status=session.get('suicide_status','not selsected')
    msg["Subject"] = (f"Mr/MS.{name}, Your Mental Health Report")
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = user_email
    if suicide_status == "No":
        msg.set_content(f"Please find your mental health report attached.{generated_response}")
    else:
        msg.set_content(f"Please Consult Gcet Student Counsellor Available at College timings as per your Conveniance.Given suggestions are given /generated based on AI.You need a Physical Counsellor support.{generated_response}")


    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
        msg.add_attachment(pdf_data, maintype="application", subtype="pdf", filename=pdf_filename)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"Email sent successfully to {user_email}")
    except Exception as e:
        print(f"Error sending email: {e}")


# ---------------------------------------------------------------------------------------------------------------------------------------------------#




def send_whatsapp_analysis(name, age, roll, year, section, branch, whatsapp_number, total_score, difficulty, suicide_status, emotion_count, highest_emotion, user_text, emotion):
    """
    Sends a mental health analysis summary via WhatsApp using Twilio API.

    Args:
        name (str): Student's name.
        whatsapp_number (str): Recipient WhatsApp number in international format (e.g., '919177041819').
        total_score (int): Total score from the mental health survey.
        difficulty (str): Level of difficulty.
        suicide_status (str): Suicide risk status.
        emotion_count (int): Count of emotions detected.
        highest_emotion (str): Most detected emotion.
        user_feeling (str): User's own description of their feeling.
        voice_emotion (str): Detected emotion from voice.
    """
    messages = []

    if not whatsapp_number:
        messages.append("❌ Error: WhatsApp number is missing.")
        return messages

    formatted_number = f"whatsapp:+{whatsapp_number}"

    message_body = (
        f"📝 *Mental Health Analysis for {name}*:\n"
        f"📝 *Age {age}*:\n"
        f"📝 *Roll Number {roll}*:\n"
        f"📝 *Year {year}*:\n"
        f"📝 *Section {section}*:\n"
        f"📝 *Branch {branch}*:\n"
        f"- *Total Score:* {total_score} / 250 → Reflects the intensity of distress from the survey.\n"
        f"- *Difficulty Level:* {difficulty} → Assessed from the responses.\n"
        f"- *Suicide Risk:* {suicide_status}\n"
        f"- *Emotion Count:* {emotion_count} → Recognized emotions from facial analysis.\n"
        f"- *Dominant Emotion:* {highest_emotion} → The most detected emotion.\n"
        f"- *Student's Self-Reported Feeling:* \"{user_text}\"\n"
        f"- *Student's Voice Emotion Detected:* \"{emotion}\""
    )

    try:
        message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,  # e.g., 'whatsapp:+14155238886'
            to=formatted_number,
            body=message_body
        )
        messages.append(f"✅ Mental health report sent to {formatted_number}")
    except Exception as e:
        messages.append(f"❌ Failed to send WhatsApp message: {str(e)}")

    return messages

# ---------------------------------------------------------------------------------------------------------------------------------------------------#



# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWISID")
TWILIO_AUTH_TOKEN = os.getenv("TWIAUTHTOKEN")
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ---------------------------------------------------------------------------------------------------------------------------------------------------#


@app.route('/convert_to_pdf', methods=['POST'])
def convert_to_pdf():
    """Generate a PDF report and optionally send via email/WhatsApp."""
    # Session data
    name = session.get('name', 'Unknown')
    age = session.get('age', 'Unknown')
    total_score = session.get('total_score', 0)
    difficulty = session.get('difficulty', 'Unknown')
    suicide_status = session.get('suicide_status', 'Unknown')
    emotion_count = session.get('emotion_count', {})
    highest_emotion = session.get('highest_emotion', 'No Data')
    user_text = session.get("user_text", "Not entered")
    emotion = session.get("emotion", "No voice data")
    user_email = session.get('user', 'Unknown')
    whatsapp_number = "919177041819"  # Can also be dynamic if needed

    roll = session.get("roll")
    year = session.get("year")
    branch = session.get("branch")
    section = session.get("section")
    pn = session.get("pn")
    messages = []

    # Check required fields
    if name == "Unknown":
        return "❌ Error: Name is required to generate PDF!", 400
    if not recipient_email or not is_valid_email(recipient_email):
        return f"❌ Error: Invalid or missing email address: {recipient_email}", 400

    # Generate PDF
    pdf_filename = f"{name}_report.pdf"
    pdf_path = os.path.join(app.root_path, 'static', 'userpdf', pdf_filename)

    html_content = render_template(
        'report.html',
        name=name,
        age=age,
        roll=roll,
        year=year,
        branch=branch,
        section=section,
        pn=pn,
        total_score=total_score,
        difficulty=difficulty,
        suicide_status=suicide_status,
        emotion_count=emotion_count,
        highest_emotion=highest_emotion,
        user_text=user_text,
        emotion=emotion
    )

    # ---------------------------------------------------------------------------------------------------------------------------------------------------#


    # Convert HTML to PDF
    pdfkit.from_string(html_content, pdf_path, configuration=pdfkit_config)

    # Email logic (only if suicide risk is yes or student email exists)
    if suicide_status == "Yes":
        send_email_with_attachment_to_gcetcounsellor(recipient_email, pdf_path, pdf_filename)
        messages.append(f"📧 PDF sent to GCET counsellor: {recipient_email}")
    
    if user_email and is_valid_email(user_email):
        send_email_with_attachment_to_gcetStudent(user_email, pdf_path, pdf_filename)
        messages.append(f"📧 PDF sent to student: {user_email}")
    else:
        messages.append("⚠️ Student email not found or invalid; PDF not sent.")

    # WhatsApp: Send report summary as plain message (no PDF)
    whatsapp_msg_result = send_whatsapp_analysis(
        name=name,
        age=age,
        roll=roll,
        year=year,
        section=section,
        branch=branch,
        whatsapp_number=whatsapp_number,
        total_score=total_score,
        difficulty=difficulty,
        suicide_status=suicide_status,
        emotion_count=emotion_count,
        highest_emotion=highest_emotion,
        user_text=user_text,
        emotion=emotion
    )
    messages.extend(whatsapp_msg_result)

    return "<br>".join(messages) 
    
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------------#


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('detected_emotions', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
