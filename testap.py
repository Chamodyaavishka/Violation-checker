from transformers import BertTokenizer
import torch
import joblib
from Testing.multi_task_bert_model1 import MultitaskBERTModel

# Define the model path
model_path = "/Users/chamodyaavishka/Desktop/NEW/Product/modelapril"

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load label encoders
source_encoder = joblib.load(f"{model_path}/source_encoder.joblib")
department_encoder = joblib.load(f"{model_path}/department_encoder.joblib")
skill_name_encoder = joblib.load(f"{model_path}/skill_name_encoder.joblib")

# Now, determine the correct counts from your model's training process for departments and skills
num_departments = len(department_encoder.classes_)
num_skills = len(skill_name_encoder.classes_)

# Initialize and load the model correctly according to the defined __init__ method in MultitaskBERTModel
model = MultitaskBERTModel(num_departments=num_departments, num_skills=num_skills)
model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
model.eval()


# Prediction function
def make_prediction(utterance):
    inputs = tokenizer(utterance, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    violation_pred = torch.argmax(outputs[0], dim=1).item()
    source_pred = torch.argmax(outputs[1], dim=1).item()
    department_pred = torch.argmax(outputs[2], dim=1).item()
    skill_name_pred = torch.argmax(outputs[3], dim=1).item()
    
    print(f"\nUtterance: {utterance}")
    print("Violation:", "Yes" if violation_pred == 1 else "No")
    print("Source:", source_encoder.inverse_transform([source_pred])[0])
    
    if source_encoder.inverse_transform([source_pred])[0] == 'Alexa Skill':
        print("Department:", department_encoder.inverse_transform([department_pred])[0])
        print("Skill Name:", skill_name_encoder.inverse_transform([skill_name_pred])[0])
    else:
        print("This is a built-in feature, no department or skill name.")

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter an utterance (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        make_prediction(user_input)
