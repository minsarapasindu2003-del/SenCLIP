import os
import shutil
import torch
import torchvision
import clip
from huggingface_hub import hf_hub_download

# 1) Official SenCLIP checkpoints (RN50)
os.makedirs("checkpoints", exist_ok=True)
for f in ["SenCLIP_AvgPool_RN50.ckpt", "SenCLIP_AttPoolPerImg_RN50.ckpt"]:
    print("Downloading", f)
    hf_hub_download("pallavijainpj/SenCLIP", f, local_dir="checkpoints")

# 2) Plain CLIP RN50 weights (baseline)
print("Saving plain CLIP RN50 baseline")
m, _ = clip.load("RN50", device="cpu")
torch.save(m.state_dict(), "checkpoints/CLIP_orig_RN50.ckpt")

# 3) EuroSAT dataset in the folder layout zeroshot.py expects
if not os.path.exists("Datasets/EuroSAT/2750"):
    print("Downloading EuroSAT")
    torchvision.datasets.EuroSAT(root="data", download=True)
    os.makedirs("Datasets", exist_ok=True)
    shutil.move("data/eurosat", "Datasets/EuroSAT")

print("EuroSAT classes:", os.listdir("Datasets/EuroSAT/2750"))
print("DONE. Checkpoints:", os.listdir("checkpoints"))
