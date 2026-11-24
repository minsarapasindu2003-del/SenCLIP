"""
Evaluate several models on EuroSAT (zero-shot) with 3 prompt sets and print one table.

Models (a row is skipped automatically if its file is missing):
  - CLIP RN50 (baseline)                : checkpoints/CLIP_orig_RN50.ckpt
  - Official SenCLIP AvgPool RN50       : checkpoints/SenCLIP_AvgPool_RN50.ckpt
  - Official SenCLIP AttPool RN50       : checkpoints/SenCLIP_AttPoolPerImg_RN50.ckpt
  - Your model (Lightning checkpoint)   : checkpoints/last.ckpt   (evaluated WITH and WITHOUT projection head)
  - Your model (converted plain ckpt)   : checkpoints/SenCLIP_mine.ckpt  (no head)

Run:  python eval_all.py --max_images 1000
"""
import os
import csv
import json
import random
import argparse

import torch
import clip
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

NEW_CNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake',
}

NORMS = {
    'clip': ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    'bing': ((0.347, 0.376, 0.296), (0.269, 0.261, 0.276)),
}

PROMPT_FILES = {
    'Aerial': 'prompts/prompt_eurosat_aerial_mixed.json',
    'Ground': 'prompts/prompt_eurosat_ground_mixed.json',
    'Ground+Aerial': 'prompts/prompt_eurosat_ground_aerial_mixed.json',
}


class ProjectionHead(nn.Module):
    """Same as ProjectionHead in SenCLIP.py (copied so this script is standalone)."""
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(d, d)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, orig):
        projected = self.projection(orig)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected + orig
        return self.layer_norm(x)


def build_model(path, kind, device):
    model, _ = clip.load("RN50", device="cpu")
    head = None
    if kind == 'plain':
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
    else:  # lightning checkpoint
        ck = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        p1, p2 = "learner.encoder.model.", "learner.projection_head."
        model.load_state_dict({k[len(p1):]: v for k, v in ck.items() if k.startswith(p1)}, strict=True)
        head = ProjectionHead(1024)
        head.load_state_dict({k[len(p2):]: v for k, v in ck.items() if k.startswith(p2)})
        head.eval()
    model = model.float().to(device).eval()
    return model, head


def make_loader(root, norm, crop, idx, bs, workers):
    mean, std = NORMS[norm]
    tfs = []
    if crop and norm == 'bing':
        tfs.append(CenterCrop(crop))
    tfs += [Resize((224, 224), interpolation=InterpolationMode.BICUBIC), ToTensor(), Normalize(mean, std)]
    ds = ImageFolder(root, transform=Compose(tfs))
    return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False, num_workers=workers), ds.classes


@torch.no_grad()
def extract_features(model, loader, device):
    feats, labels = [], []
    for x, y in tqdm(loader, desc="  images"):
        feats.append(model.encode_image(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def class_prototypes(model, class_names, templates, device):
    protos = []
    for c in class_names:
        toks = clip.tokenize(templates[c], truncate=True).to(device)
        e = model.encode_text(toks).float()
        e = e / e.norm(dim=-1, keepdim=True)
        m = e.mean(0)
        protos.append(m / m.norm())
    return torch.stack(protos).cpu()


def top1(feats, labels, protos):
    f = feats / feats.norm(dim=-1, keepdim=True)
    logits = f @ protos.T
    logits = logits / (logits.mean(0, keepdim=True) + 1e-6)   # same calibration as zeroshot.py
    return (logits.argmax(1) == labels).float().mean().item() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', default='.')
    ap.add_argument('--max_images', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--crop', type=int, default=0,
                    help="Optional: center-crop EuroSAT to this size before resizing (SenCLIP models only). 0 = off")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    root = os.path.join(args.root_dir, 'Datasets/EuroSAT/2750/')

    n_total = len(ImageFolder(root))
    random.seed(0)
    idx = random.sample(range(n_total), min(args.max_images, n_total)) if args.max_images else list(range(n_total))
    print(f"Evaluating on {len(idx)} of {n_total} images (seed 0, same images for every model)")

    templates = {k: json.load(open(v)) for k, v in PROMPT_FILES.items()}

    specs = [
        ("CLIP RN50 (baseline)",       "checkpoints/CLIP_orig_RN50.ckpt",              "plain",     "clip"),
        ("Official SenCLIP AvgPool",   "checkpoints/SenCLIP_AvgPool_RN50.ckpt",        "plain",     "bing"),
        ("Official SenCLIP AttPool",   "checkpoints/SenCLIP_AttPoolPerImg_RN50.ckpt",  "plain",     "bing"),
        ("Mine",                       "checkpoints/last.ckpt",                        "lightning", "bing"),
        ("Mine (converted, no head)",  "checkpoints/SenCLIP_mine.ckpt",                "plain",     "bing"),
    ]

    rows = []
    for name, path, kind, norm in specs:
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue
        print(f"\n=== {name} ===")
        model, head = build_model(path, kind, device)
        loader, classes = make_loader(root, norm, args.crop, idx, args.batch_size, args.num_workers)
        class_names = [NEW_CNAMES[c] for c in classes]
        feats, labels = extract_features(model, loader, device)

        variants = [(name + (" (no head)" if head is not None else ""), feats)]
        if head is not None:
            with torch.no_grad():
                variants.append((name + " (WITH head)", head(feats)))

        protos = {}
        for pname, tpl in templates.items():
            protos[pname] = class_prototypes(model, class_names, tpl, device)

        for vname, vfeats in variants:
            res = {p: top1(vfeats, labels, protos[p]) for p in PROMPT_FILES}
            rows.append((vname, res))
            print("  ", vname, {k: round(v, 2) for k, v in res.items()})

    print("\n" + "=" * 70)
    print(f"{'Model':38s}" + "".join(f"{p:>14s}" for p in PROMPT_FILES))
    for vname, res in rows:
        print(f"{vname:38s}" + "".join(f"{res[p]:14.2f}" for p in PROMPT_FILES))
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + list(PROMPT_FILES))
        for vname, res in rows:
            w.writerow([vname] + [round(res[p], 2) for p in PROMPT_FILES])
    print("\nSaved results.csv")


if __name__ == "__main__":
    main()
