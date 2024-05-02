import io
import torch
import numpy as np
import joblib
from transformers import BertTokenizer
from pydub import AudioSegment
import soundfile as sf
import speech_recognition as sr
import gradio as gr
import logging

# Custom model import
from multi_task_bert_model import MultitaskBERTModel

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Paths configuration
model_directory_path = '/Users/chamodyaavishka/Desktop/NEW/Testing/new_model'

class Predictor:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        num_departments = 13
        num_skills = 102
        self.model = MultitaskBERTModel(num_departments=num_departments, num_skills=num_skills)
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(model_state_dict, strict=False)
        self.model.eval()

        self.source_le = joblib.load(f"{model_directory_path}/Source_label_encoder.joblib")
        self.department_le = joblib.load(f"{model_directory_path}/Department_label_encoder.joblib")
        self.skill_name_le = joblib.load(f"{model_directory_path}/Skill_Name_label_encoder.joblib")

    def predict_from_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        # Ensure all outputs are properly handled
        if len(outputs) < 4:
            logging.error("Insufficient model outputs.")
            return {"Error": "Insufficient outputs from the model"}

        return {
            'violation': 'Yes' if torch.argmax(outputs[0], dim=1).item() == 1 else 'No',
            'source': self.source_le.inverse_transform([torch.argmax(outputs[1], dim=1).item()])[0],
            'department': self.department_le.inverse_transform([torch.argmax(outputs[2], dim=1).item()])[0],
            'skill_name': self.skill_name_le.inverse_transform([torch.argmax(outputs[3], dim=1).item()])[0]
        }

def speech_to_text(audio_data):
    recognizer = sr.Recognizer()
    if not audio_data or len(audio_data) < 2:
        return "Invalid audio data"
    
    audio_samples, sample_rate = audio_data
    
    try:
        with io.BytesIO() as audio_buffer:
            sf.write(audio_buffer, audio_samples, sample_rate, format='wav')
            audio_buffer.seek(0)
            with sr.AudioFile(audio_buffer) as source:
                audio_recorded = recognizer.record(source)
                text = recognizer.recognize_google(audio_recorded)
    except sr.UnknownValueError:
        return "Speech was unclear"
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    except Exception as e:
        return f"Error processing audio data: {str(e)}"
    return text
def gradio_predict(input_audio):
    if input_audio is None:
        return "No audio input received"
    transcribed_text = speech_to_text(input_audio)
    if transcribed_text in ["Speech was unclear", "Could not request results"]:
        return transcribed_text
    predictions = predictor.predict_from_text(transcribed_text)
    response_text = f"Violation: {predictions['violation']}\nSource: {predictions['source']}\nDepartment: {predictions['department']}\nSkill Name: {predictions['skill_name']}"
    return response_text

# Initialize the predictor
predictor = Predictor(model_directory_path + '/multi_task_bert_model.bin')

# Set up the Gradio interface
iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Audio(type="numpy", label="Record or Upload Audio"),
    outputs=gr.Textbox(label="Model Prediction"),
    analytics_enabled=False  # Disable Gradio's analytics to prevent unnecessary network requests
)

if __name__ == "__main__":
    iface.launch(share=True)
