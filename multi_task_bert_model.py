import torch
import torch.nn as nn
from transformers import BertModel

class MultitaskBERTModel(nn.Module):
    def __init__(self, num_departments, num_skills, bert_model='bert-base-uncased'):
        super(MultitaskBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.1)
        
        # Updated layer names
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # For violation detection
        self.source_classifier = nn.Linear(self.bert.config.hidden_size, 2)  # For source classification
        self.department_classifier = nn.Linear(self.bert.config.hidden_size, num_departments)  # For department classification
        self.skill_name_classifier = nn.Linear(self.bert.config.hidden_size, num_skills)  # For skill name classification


    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        
        # Use the updated attribute names
        violation_pred = self.classifier(pooled_output)
        source_pred = self.source_classifier(pooled_output)
        department_pred = self.department_classifier(pooled_output)
        skill_name_pred = self.skill_name_classifier(pooled_output)
        
        return violation_pred, source_pred, department_pred, skill_name_pred
