import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical federated learning
    
    FL(p_t) = -α(1-p_t)^γ * log(p_t)
    
    Designed for binary classification with severe class imbalance (e.g., medical mortality prediction)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for positive class
        self.gamma = gamma  # Focusing parameter to down-weight easy examples
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits [N] or [N, 1] for binary classification
            targets: Ground truth binary labels [N]
        """
        if inputs.dim() > 1 and inputs.size(-1) == 1:
            inputs = inputs.squeeze(-1)
        
        targets = targets.float()
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t
        p_t = torch.exp(-bce_loss)
        
        # Compute alpha term  
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



def create_loss(args, device=None, **kwargs):
    if "client_index" in kwargs:
        client_index = kwargs["client_index"]
    else:
        client_index = args.client_index

    if args.loss_fn == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_fn == "nll_loss":
        loss_fn = nn.NLLLoss()
    else:
        raise NotImplementedError

    return loss_fn















