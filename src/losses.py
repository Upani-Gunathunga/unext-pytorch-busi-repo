import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()  # applies sigmoid internally, then BCE

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        # Dice loss: measure overlap between predicted and true regions
        pred_prob = torch.sigmoid(pred)  # squash raw outputs to 0-1 range
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + 1e-7) / (pred_flat.sum() + target_flat.sum() + 1e-7)
        dice_loss = 1 - dice_score

        return 0.5 * bce_loss + dice_loss