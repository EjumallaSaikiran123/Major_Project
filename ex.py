import os
from keras.models import load_model

model_path = os.path.abspath("models/voice_model.h5")
print("Loading model from:", model_path)
model = load_model(model_path)
