import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class WordPredictionTransformer(nn.Module):
    def __init__(self, 
                 model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 feature_dim : int = 1024,  # Adjust based on your combined feature dimensions
                 vocab_size : int = 30522,  # Default BERT vocab size
                 max_length : int = 512,
                 dropout : float = 0.1):

        super().__init__()
        
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=vocab_size
        )
        
        self.feature_projection = nn.Linear(feature_dim, self.transformer.config.hidden_size)
        
        self.word_prediction_head = nn.Linear(
            self.transformer.config.hidden_size, 
            vocab_size
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.max_length = max_length


    def forward(self, combined_features):
        projected_features = self.feature_projection(combined_features)   
        projected_features = self.dropout(projected_features)   
        word_prediction_logits = self.word_prediction_head(projected_features)    
        return word_prediction_logits

    def predict_next_words(self, combined_features, top_k=5):
        logits = self.forward(combined_features) 
        probabilities = torch.softmax(logits, dim=-1)        
        top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k, dim=-1)    
        return list(zip(top_k_indices.tolist(), top_k_probs.tolist()))




"""combined_features = torch.randn(1, 1024)  # Batch size of 1, feature dim of 1024
    
word_predictor = WordPredictionTransformer()
    
predictions = word_predictor.predict_next_words(combined_features)
    
print("Top 5 predicted words:")
for idx, prob in predictions:
    print(f"Word Index: {idx}, Probability: {prob}")"""
