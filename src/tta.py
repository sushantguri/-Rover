"""
Test-Time Augmentation (TTA) Multi-View Ensemble Engine.
Boosts segmentation precision and boundary stability across harsh Martian lighting conditions
by evaluating multiple geometric/photometric views and averaging aligned softmax probabilities.
"""
import torch
import torch.nn.functional as F
from typing import Optional


class TestTimeAugmenter:
    """
    Performs multi-view Test-Time Augmentation (TTA) for semantic segmentation models.
    Supported modes:
      - 'fast': 2 views (Original + Horizontal Flip)
      - 'full': 4 views (Original + H-Flip + Shadow Boost + Highlight Attenuation)
    """
    def __init__(self, model: torch.nn.Module, mode: str = "fast"):
        self.model = model
        self.mode = mode

    @torch.no_grad()
    def predict_probs(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Runs TTA forward passes and returns ensembled probability map of shape (B, C, H, W).
        """
        self.model.eval()
        probs_list = []

        # View 1: Original
        logits_orig = self.model(img_tensor)
        probs_orig = F.softmax(logits_orig, dim=1)
        probs_list.append(probs_orig)

        # View 2: Horizontal Flip (dim=3 is Width)
        flipped_input = torch.flip(img_tensor, dims=[3])
        logits_flip = self.model(flipped_input)
        probs_flip = F.softmax(logits_flip, dim=1)
        # Un-flip probabilities back to original orientation
        probs_flip_unflipped = torch.flip(probs_flip, dims=[3])
        probs_list.append(probs_flip_unflipped)

        if self.mode == "full":
            # View 3: Brightened / Shadow recovery (subtle 10% intensity boost in normalized space)
            bright_input = img_tensor * 1.10 + 0.05
            logits_bright = self.model(bright_input)
            probs_list.append(F.softmax(logits_bright, dim=1))

            # View 4: Attenuated highlights (glare reduction)
            dim_input = img_tensor * 0.90 - 0.05
            logits_dim = self.model(dim_input)
            probs_list.append(F.softmax(logits_dim, dim=1))

        # Ensemble average across views
        ensemble_probs = torch.mean(torch.stack(probs_list, dim=0), dim=0)
        return ensemble_probs

    @torch.no_grad()
    def predict_pseudo_logits(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Returns log-probabilities compatible with downstream entropy/argmax operations.
        """
        probs = self.predict_probs(img_tensor)
        # Clamp for numerical stability before log
        eps = 1e-7
        log_probs = torch.log(torch.clamp(probs, min=eps, max=1.0))
        return log_probs
