import torch

from src.models.hybrid_model import ASCIIHybridDeepfakeDetector


def test_hybrid_forward_shape():
    model = ASCIIHybridDeepfakeDetector({"device": "cpu", "model": {"pretrained": False, "feature_dim": 128}})
    pixel = torch.randn(2, 8, 3, 256, 256)
    ascii_in = torch.randn(2, 8, 3, 80, 40)
    out = model(pixel, ascii_in)
    assert out.shape == (2,)
