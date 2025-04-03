import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from midas_depth_estimator import MiDaSDepthEstimator
from resnet_classifier import ResNetDepthClassifier
from mlp_classifier import MLPDepthClassifier

class PipelineBModel(nn.Module):
    """
    Pipeline B: RGB input -> MiDaS depth map -> ResNet-18 -> Table/No-Table
    """
    def __init__(self, midas_model=None, classifier=None, freeze_midas=True):
        super().__init__()
        self.midas = midas_model if midas_model is not None else MiDaSDepthEstimator()
        self.classifier = classifier if classifier is not None else MLPDepthClassifier()

        # freeze MiDaS model
        if freeze_midas and self.midas is not None:
            for param in self.midas.model.parameters():
                param.requires_grad = False

    def forward(self, rgb):  # rgb: [B,3,H,W]
        batch_size = rgb.shape[0]
        device = rgb.device
        
        # handle batch (MiDaS predict only do single image)
        depths = []
        for i in range(batch_size):
            # tensor to numpy for MiDaS
            single_rgb = rgb[i].permute(1, 2, 0).cpu().numpy()  # [H,W,3]
            # make sure [0, 255] range
            if single_rgb.max() <= 1.0:
                single_rgb = (single_rgb * 255).astype(np.uint8)
            
            # get depth map
            depth = self.midas.predict(single_rgb)  # [H,W]
            depths.append(depth)
        
        # stack depth maps to a batch
        depth_batch = torch.stack(depths).to(device)  # [B,H,W]
        # print(f"batch shape: {depth_batch.shape}")

        # to classifier
        logits = self.classifier(depth_batch)  # [B,2]
        
        return logits # [B,2]
    
    def predict(self, rgb):
        with torch.no_grad():
            logits = self.forward(rgb)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        return predictions, probs


if __name__ == "__main__": # test
    midas = MiDaSDepthEstimator(model_path="src/pipelineB/weights/dpt_large_384.pt")
    classifier = MLPDepthClassifier()
    model = PipelineBModel(midas, classifier)

    batch_rgb = torch.randint(0, 255, (4, 3, 480, 640), dtype=torch.float32) / 255.0  # fake input

    batch_predictions, batch_probs = model.predict(batch_rgb)
    print(f"output: {batch_predictions}")
    for i, (pred, prob) in enumerate(zip(batch_predictions, batch_probs)):
        print(f"image {i}: pred = {pred.item()}, prob = {prob}") # prob = [no_table, has_table]
