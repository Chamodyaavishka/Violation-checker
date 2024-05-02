import gradio as gr
from transformers import BertTokenizer
import torch
import joblib
from Testing.multi_task_bert_model1 import MultitaskBERTModel
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import os

# Define the model path and load components
model_path = "/Users/chamodyaavishka/Desktop/NEW/Product/modelapril"
tokenizer = BertTokenizer.from_pretrained(model_path)
source_encoder = joblib.load(f"{model_path}/source_encoder.joblib")
department_encoder = joblib.load(f"{model_path}/department_encoder.joblib")
skill_name_encoder = joblib.load(f"{model_path}/skill_name_encoder.joblib")
num_departments = len(department_encoder.classes_)
num_skills = len(skill_name_encoder.classes_)
model = MultitaskBERTModel(num_departments=num_departments, num_skills=num_skills)
model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
model.eval()

# Speech recognition and TTS
r = sr.Recognizer()

def voice_to_text(audio_file):
    with sr.AudioFile(audio_file.name) as source:
        audio_data = r.record(source)
    return r.recognize_google(audio_data)

def text_to_voice(text):
    tts = gTTS(text)
    byte_io = BytesIO()
    tts.write_to_fp(byte_io)
    return byte_io.getvalue()

# Prediction function adapted for voice
def make_prediction(audio_file):
    utterance = voice_to_text(audio_file)
    inputs = tokenizer(utterance, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    violation_pred = torch.argmax(outputs[0], dim=1).item()
    source_pred = torch.argmax(outputs[1], dim=1).item()
    response_text = f"Utterance: {utterance}\nViolation: {'Yes' if violation_pred == 1 else 'No'}\n"
    response_text += f"Source: {source_encoder.inverse_transform([source_pred])[0]}"
    if source_encoder.inverse_transform([source_pred])[0] == 'Alexa Skill':
        department_pred = torch.argmax(outputs[2], dim=1).item()
        skill_name_pred = torch.argmax(outputs[3], dim=1).item()
        response_text += f"\nDepartment: {department_encoder.inverse_transform([department_pred])[0]}"
        response_text += f"\nSkill Name: {skill_name_encoder.inverse_transform([skill_name_pred])[0]}"
    else:
        response_text += "\nThis is a built-in feature, no department or skill name."
    return text_to_voice(response_text)

# Set up the Gradio interface
iface = gr.Interface(
    fn=make_prediction,
    inputs=gr.inputs.Audio(source="microphone", type="file"),
    outputs="audio",
    title="Multitask Model Prediction"
)

# Launch the application
iface.launch()
