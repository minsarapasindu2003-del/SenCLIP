import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from zeroshot import load_model, transform_img

FEATURES = SCRIPT_DIR / "features"
MEAN, STD = (0.347, 0.376, 0.296), (0.269, 0.261, 0.276)

CLASS_NAME_MAPPING = {
    "AnnualCrop": "Annual Crop Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "River": "River",
    "SeaLake": "Sea or Lake",
}


def main():
    p = argparse.ArgumentParser(description="Predict land-use class with SenCLIP + paper prompt selection + DAP")
    p.add_argument("images", nargs="+", help="image path(s)")
    p.add_argument("--k", type=int, default=5, help="prompts kept per class")
    p.add_argument("--ckpt_path", default=str(PROJECT_ROOT / "SenCLIP_AvgPool_ViTB32.ckpt"))
    p.add_argument("--model_arch", default="ViT-B/32")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt_path, args.model_arch, device)

    img_data = torch.load(FEATURES / "features_senclip_avgpool.pt", map_location="cpu")
    txt_data = torch.load(FEATURES / "features_text_senclip_avgpool.pt", map_location="cpu")
    sel = torch.load(FEATURES / "prompt_selection_results.pt", map_location="cpu")

    if args.k not in sel["top_k_values"]:
        raise ValueError(f"K must be one of {sel['top_k_values']}")

    X = img_data["features"].float()
    T = txt_data["features"].float()
    class_names = txt_data["class_names"]
    idx = sel["best_indices"][args.k]
    C = T.shape[0]

    # Selected prompts per class, and p(a) from the whole image set
    selected = [T[c, idx[c]] for c in range(C)]
    priors = [(X @ s.T / 2 + 1).mean(dim=0) for s in selected]

    transform = transform_img(MEAN, STD)

    for path in args.images:
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model.encode_image(x).float().cpu()
            f = f / f.norm(dim=-1, keepdim=True)

        scores = torch.stack([
            torch.log((f @ selected[c].T / 2 + 1) / priors[c]).sum()
            for c in range(C)
        ])
        top = scores.topk(3)

        true = CLASS_NAME_MAPPING.get(Path(path).parent.name)
        pred = class_names[top.indices[0]]
        mark = "" if true is None else ("  [correct]" if pred == true else f"  [wrong, true: {true}]")

        print(f"\nImage: {Path(path).name}")
        print(f"Predicted: {pred}{mark}")
        for r, (i, s) in enumerate(zip(top.indices, top.values), start=1):
            print(f"  {r}. {class_names[i]}  (score {s.item():.2f})")


if __name__ == "__main__":
    main()