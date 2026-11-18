import torch
from pathlib import Path

D = Path(__file__).resolve().parent / "features"
img = torch.load(D / "features_senclip_avgpool.pt", map_location="cpu")
txt = torch.load(D / "features_text_senclip_avgpool.pt", map_location="cpu")

X, y = img["features"], img["labels"]
print("Image features:", X.shape, "| labels:", y.shape)

W = txt["features"].mean(dim=1)                      # average the 46 prompts per class
W = W / W.norm(dim=-1, keepdim=True)

logits = X @ W.T
logits = logits / (logits.mean(0, keepdim=True) + 1e-6)   # same step as zeroshot.py

print("Top-1:", round((logits.argmax(1) == y).float().mean().item() * 100, 2))
print("Top-5:", round((logits.topk(5, 1).indices == y[:, None]).any(1).float().mean().item() * 100, 2))