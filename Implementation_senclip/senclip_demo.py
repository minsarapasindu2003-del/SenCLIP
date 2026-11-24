"""
SenCLIP demo: satellite image(s) + your target classes
  -> best ground-view and aerial-view prompts per class, ranked with the paper's
     prompt-selection score (Eq. 6-7: w = alpha / beta), for every available model
  -> predicted class for each image, using the selected prompts.

Examples:
  python senclip_demo.py --image_dir my_image --ground_json prompts/my_ground.json --aerial_json prompts/my_aerial.json
  python senclip_demo.py --image my_image/airport_1.png --ground_json prompts/my_ground.json --aerial_json prompts/my_aerial.json
"""
import os
import re
import json
import argparse

import torch
import clip
from torch import nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

MEAN, STD = (0.347, 0.376, 0.296), (0.269, 0.261, 0.276)   # normalization used by SenCLIP training


class ProjectionHead(nn.Module):
    """Same as ProjectionHead in SenCLIP.py (only used for the Lightning checkpoint of your own model)."""
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(d, d)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, orig):
        projected = self.projection(orig)
        x = self.fc(self.gelu(projected))
        x = self.dropout(x)
        return self.layer_norm(x + projected + orig)


def find_ckpt(*words):
    """Find a file in checkpoints/ whose lower-case name contains all the given words."""
    if not os.path.isdir("checkpoints"):
        return None
    for f in sorted(os.listdir("checkpoints")):
        n = f.lower()
        if n.endswith(".ckpt") and all(w in n for w in words):
            return os.path.join("checkpoints", f)
    return None


def model_list():
    specs = [
        ("RN50 AvgPool (official)",     find_ckpt("avgpool", "rn50"), "RN50",     "plain"),
        ("RN50 AttPool (official)",     find_ckpt("attpool", "rn50"), "RN50",     "plain"),
        ("ViT-B/32 AvgPool (official)", find_ckpt("avgpool", "vit"),  "ViT-B/32", "plain"),
        ("ViT-B/32 AttPool (official)", find_ckpt("attpool", "vit"),  "ViT-B/32", "plain"),
    ]
    if os.path.exists("checkpoints/last.ckpt"):
        specs.append(("My RN50 AvgPool (trained by me)", "checkpoints/last.ckpt", "RN50", "lightning"))
    elif os.path.exists("checkpoints/SenCLIP_mine.ckpt"):
        specs.append(("My RN50 AvgPool (trained by me)", "checkpoints/SenCLIP_mine.ckpt", "RN50", "plain"))
    return specs


def load_model(path, arch, kind):
    model, _ = clip.load(arch, device="cpu")
    head = None
    if kind == "plain":
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        p1, p2 = "learner.encoder.model.", "learner.projection_head."
        model.load_state_dict({k[len(p1):]: v for k, v in sd.items() if k.startswith(p1)}, strict=True)
        head = ProjectionHead(model.visual.output_dim)
        head.load_state_dict({k[len(p2):]: v for k, v in sd.items() if k.startswith(p2)})
        head.eval()
    return model.float().eval(), head


@torch.no_grad()
def encode_prompts(model, prompts, classes, T):
    """-> tensor (C, T, D) of normalized text embeddings."""
    out = []
    for c in classes:
        e = model.encode_text(clip.tokenize(prompts[c][:T], truncate=True)).float()
        out.append(e / e.norm(dim=-1, keepdim=True))
    return torch.stack(out)


def prompt_scores(E):
    """
    Paper Eq. 6-7.  E: (C, T, D) normalized prompt embeddings.
      alpha[c,t] = mean similarity of prompt (c,t) to the prompts of its own class c      (Eq. 6)
      beta[c,t]  = mean similarity of prompt (c,t) to ALL prompts of ALL classes          (Eq. 7)
      w = alpha / beta   (higher = more representative of its class)
    """
    C, T, D = E.shape
    flat = E.reshape(C * T, D)
    S = flat @ flat.T                                   # (CT, CT) prompt-prompt similarity
    beta = (S.sum(1) / (C * T)).reshape(C, T)
    S4 = S.reshape(C, T, C, T)
    alpha = torch.stack([S4[c, :, c, :].sum(1) for c in range(C)]) / T
    return alpha / beta


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def collect_images(args):
    if args.image_dir:
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        files = sorted(os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.lower().endswith(exts))
    elif args.image:
        files = [args.image]
    else:
        raise SystemExit("Give --image_dir <folder> or --image <file>")
    if not files:
        raise SystemExit("No images found")
    return files


def expected_class(path, classes):
    """'airport_1.png' -> class whose name matches 'airport' (ignoring case, spaces, underscores); else None."""
    stem = os.path.splitext(os.path.basename(path))[0]
    prefix = re.sub(r"[_\-\s]*\d+$", "", stem)
    for c in classes:
        if norm(c) == norm(prefix):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", default="", help="folder with your images (png/jpg/tif)")
    ap.add_argument("--image", default="", help="a single image file")
    ap.add_argument("--classes", default="", help="comma separated target classes (default: all classes in the prompt file)")
    ap.add_argument("--ground_json", default="prompts/my_ground.json")
    ap.add_argument("--aerial_json", default="prompts/my_aerial.json")
    ap.add_argument("--topk", type=int, default=5, help="number of best prompts kept per class")
    args = ap.parse_args()

    views = {"GROUND": json.load(open(args.ground_json, encoding="utf-8")),
             "AERIAL": json.load(open(args.aerial_json, encoding="utf-8"))}
    all_classes = list(views["GROUND"].keys())
    missing = [c for c in all_classes if c not in views["AERIAL"]]
    if missing:
        raise SystemExit(f"These classes are in the ground file but not in the aerial file: {missing}")

    if args.classes.strip():
        lookup = {c.lower(): c for c in all_classes}
        classes = []
        for c in [x.strip() for x in args.classes.split(",") if x.strip()]:
            if c.lower() not in lookup:
                raise SystemExit(f"Class '{c}' not in prompt file. Available: {all_classes}")
            classes.append(lookup[c.lower()])
    else:
        classes = all_classes
    T = min(len(v[c]) for v in views.values() for c in classes)
    K = min(args.topk, T)

    files = collect_images(args)
    expected = [expected_class(f, classes) for f in files]
    tf = Compose([Resize((224, 224), interpolation=InterpolationMode.BICUBIC), ToTensor(), Normalize(MEAN, STD)])
    imgs = torch.stack([tf(Image.open(f).convert("RGB")) for f in files])

    lines = []
    lines.append(f"Images ({len(files)}): {[os.path.basename(f) for f in files]}")
    lines.append(f"Target classes ({len(classes)}): {classes}")
    lines.append(f"Prompts per class per view: {T}; kept per class: top-{K} by w = alpha/beta (paper Eq. 6-7)")
    lines.append("Note: the prompt ranking does not depend on the image, so it is shown once per model.")

    preds = {}          # (model label, view) -> list of (class, prob) per image
    loaded = []
    for label, path, arch, kind in model_list():
        lines.append("\n" + "=" * 78)
        lines.append(f"MODEL: {label}")
        if path is None or not os.path.exists(path):
            lines.append("  [skipped: checkpoint not found in checkpoints/]")
            continue
        try:
            model, head = load_model(path, arch, kind)
        except Exception as e:
            lines.append(f"  [skipped: could not load {path}: {type(e).__name__}: {str(e)[:150]}]")
            continue
        loaded.append(label)

        with torch.no_grad():
            f = model.encode_image(imgs).float()
            if head is not None:
                f = head(f)
            f = f / f.norm(dim=-1, keepdim=True)

        for vname, prompts in views.items():
            E = encode_prompts(model, prompts, classes, T)          # (C, T, D)
            w = prompt_scores(E)                                    # (C, T)
            top = w.topk(K, dim=1).indices                          # (C, K)

            protos = torch.stack([E[c, top[c]].mean(0) for c in range(len(classes))])
            protos = protos / protos.norm(dim=-1, keepdim=True)
            probs = (100.0 * f @ protos.T).softmax(-1)              # (N images, C)
            preds[(label, vname)] = [(classes[p.argmax().item()], p.max().item()) for p in probs]

            lines.append(f"\n  [{vname} view] top-{K} prompts per class (highest w first)")
            for ci, c in enumerate(classes):
                lines.append(f"    {c}:")
                for r, t in enumerate(top[ci].tolist(), 1):
                    lines.append(f"      {r}. [w={w[ci, t]:.3f}] {prompts[c][t]}")

    # ---------------- predictions per image ----------------
    lines.append("\n" + "=" * 78)
    lines.append("PREDICTIONS PER IMAGE (class, probability)   [OK/WRONG shown when the file name matches a class]")
    for i, fpath in enumerate(files):
        exp = expected[i]
        lines.append(f"\n{os.path.basename(fpath)}   (expected from file name: {exp if exp else 'unknown'})")
        for label in loaded:
            parts = []
            for vname in views:
                c, p = preds[(label, vname)][i]
                mark = "" if exp is None else ("  OK" if c == exp else "  WRONG")
                parts.append(f"{vname.lower()}: {c} ({p:.2f}){mark}")
            lines.append(f"  {label:34s} " + " | ".join(parts))

    if any(e is not None for e in expected):
        lines.append("\n" + "=" * 78)
        lines.append("ACCURACY on images whose file name matches a class")
        idx = [i for i, e in enumerate(expected) if e is not None]
        for label in loaded:
            parts = []
            for vname in views:
                ok = sum(preds[(label, vname)][i][0] == expected[i] for i in idx)
                parts.append(f"{vname.lower()}: {ok}/{len(idx)}")
            lines.append(f"  {label:34s} " + " | ".join(parts))

    text = "\n".join(lines)
    print(text)
    out = "results_" + (os.path.basename(os.path.normpath(args.image_dir)) if args.image_dir
                        else os.path.splitext(os.path.basename(args.image))[0]) + ".txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

    