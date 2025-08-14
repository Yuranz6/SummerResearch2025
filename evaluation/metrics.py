import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, f1_score, log_loss, average_precision_score

class MedicalMetrics:
    """
    Medical evaluation metrics for imbalanced eICU mortality prediction
    """
    
    @staticmethod
    def calculate_auprc(y_true, y_pred_proba):
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            return auc(recall, precision)
        except:
            return np.nan
    
    @staticmethod 
    def calculate_accuracy(y_true, y_pred_proba, threshold=0.5):
        y_pred = (y_pred_proba > threshold).astype(int)
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def calculate_f1_score(y_true, y_pred_proba, threshold=0.5):
        y_pred = (y_pred_proba > threshold).astype(int)
        return f1_score(y_true, y_pred, zero_division=0)
    
    @staticmethod
    def calculate_bce_loss(y_true, y_pred_proba):
        try:
            return log_loss(y_true, y_pred_proba, eps=1e-15)
        except:
            return np.nan
    
    @staticmethod
    def evaluate_model_predictions(y_true, y_pred_proba):
        return {
            'auprc': MedicalMetrics.calculate_auprc(y_true, y_pred_proba),
            'accuracy': MedicalMetrics.calculate_accuracy(y_true, y_pred_proba),
            'f1_score': MedicalMetrics.calculate_f1_score(y_true, y_pred_proba),
            'loss': MedicalMetrics.calculate_bce_loss(y_true, y_pred_proba)
        }
    
    @staticmethod
    def evaluate_model(model, x_test, y_test, device='cuda'):
        """
        Evaluate a trained model on test data
        """
        model.eval()
        with torch.no_grad():
            if not isinstance(x_test, torch.Tensor):
                x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
            if not isinstance(y_test, torch.Tensor):
                y_test = torch.tensor(y_test, dtype=torch.float32)
            
            outputs = model(x_test)
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze()
            
            y_pred_proba = torch.sigmoid(outputs).cpu().numpy()
            y_true = y_test.cpu().numpy()
            
            return MedicalMetrics.evaluate_model_predictions(y_true, y_pred_proba)