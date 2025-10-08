import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class Medical_MLP_Classifier(nn.Module):
    """
    Architecture:Fully connected layers with batch normalization and dropout
    """
    
    def __init__(self, input_dim=268, num_classes=1, hidden_dims=[128, 64],
                 dropout_rate=0.2, use_batch_norm=False):
        
        super(Medical_MLP_Classifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU(inplace=True))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        self._initialize_weights()
        
        logging.info(f"Medical_MLP_Classifier initialized: input_dim={input_dim}, "
                    f"hidden_dims={hidden_dims}, num_classes={num_classes}, "
                    f"dropout_rate={dropout_rate}")
    
    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        logits = self.classifier(x)
        return logits
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            
            if self.num_classes == 1:
                probabilities = torch.sigmoid(logits)
            else:
                probabilities = F.softmax(logits, dim=1)
        
        return probabilities
    
    def predict(self, x, threshold=0.5):
        probabilities = self.predict_proba(x)
        
        return torch.argmax(probabilities, dim=1).float()