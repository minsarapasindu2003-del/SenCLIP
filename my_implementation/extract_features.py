
import argparse
import os
import sys
from pathlib import Path

import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# --------------------------------------------------
# Project paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.append(str(PROJECT_ROOT / "src"))

from zeroshot import load_model, transform_img


def main():
    p = argparse.ArgumentParser()

    # Dataset and model paths
    p.add_argument(
        "--root_dir",
        default=str(PROJECT_ROOT)
    )
    p.add_argument(
        "--ckpt_path",
        default=str(PROJECT_ROOT / "SenCLIP_AvgPool_ViTB32.ckpt")
    )

    p.add_argument("--model_arch", default="ViT-B/32")
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Use plain CLIP instead of SenCLIP"
    )

    # Output path
    p.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "features" / "features_senclip_avgpool.pt")
    )

    p.add_argument("--batch_size", type=int, default=16)

    args = p.parse_args()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    if args.baseline:
        mean, std = (
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )

        model, _ = clip.load(args.model_arch, device=device)
        model.eval()

    else:
        mean, std = (
            (0.347, 0.376, 0.296),
            (0.269, 0.261, 0.276)
        )

        model = load_model(
            args.ckpt_path,
            args.model_arch,
            device
        )

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    dataset_path = Path(args.root_dir) / "Datasets" / "EuroSAT" / "2750"

    ds = ImageFolder(
        str(dataset_path),
        transform=transform_img(mean, std)
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Number of images: {len(ds)}")
    print(f"Number of classes: {len(ds.classes)}")

    # --------------------------------------------------
    # Extract image features
    # --------------------------------------------------

    feats, labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Encoding images"):

            f = model.encode_image(x.to(device)).float()

            # L2 normalization
            f /= f.norm(dim=-1, keepdim=True)

            feats.append(f.cpu())
            labels.append(y)

    # --------------------------------------------------
    # Save extracted features
    # --------------------------------------------------

    output_path = Path(args.out).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save({
        "features": torch.cat(feats),
        "labels": torch.cat(labels),
        "paths": [s[0] for s in ds.samples],
        "class_folders": ds.classes,
    }, output_path)

    print(f"\nSaved features to: {output_path}")
    print(f"Feature shape: {torch.cat(feats).shape}")
    print(f"Label shape: {torch.cat(labels).shape}")


if __name__ == "__main__":
    main()
