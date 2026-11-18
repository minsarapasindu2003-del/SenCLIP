
import json
import sys
from pathlib import Path

import torch
import clip
from tqdm import tqdm
from torchvision.datasets import ImageFolder

# --------------------------------------------------
# Project paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.append(str(PROJECT_ROOT / "src"))

from zeroshot import load_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_ARCH = "ViT-B/32"

CHECKPOINT_PATH = PROJECT_ROOT / "SenCLIP_AvgPool_ViTB32.ckpt"

PROMPT_PATH = (
    PROJECT_ROOT
    / "src"
    / "prompts"
    / "prompt_eurosat_aerial_mixed.json"
)

DATASET_PATH = PROJECT_ROOT / "Datasets" / "EuroSAT" / "2750"

OUTPUT_PATH = (
    SCRIPT_DIR
    / "features"
    / "features_text_senclip_avgpool.pt"
)


# Map ImageFolder class names to JSON prompt class names
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

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Load SenCLIP model
    # --------------------------------------------------

    print("\nLoading SenCLIP model...")

    model = load_model(
        str(CHECKPOINT_PATH),
        MODEL_ARCH,
        device
    )

    model.eval()

    # --------------------------------------------------
    # Load prompt JSON
    # --------------------------------------------------

    print("\nLoading prompts...")

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        templates = json.load(f)

    # Get actual EuroSAT class order from ImageFolder
    dataset = ImageFolder(str(DATASET_PATH))

    class_names = [
        CLASS_NAME_MAPPING[class_folder]
        for class_folder in dataset.classes
    ]

    # Check that all required prompt categories exist
    for class_name in class_names:
        if class_name not in templates:
            raise ValueError(
                f"Missing prompts for class: {class_name}"
            )

    num_classes = len(class_names)
    num_prompts = len(templates[class_names[0]])

    print(f"Number of classes: {num_classes}")
    print(f"Prompts per class: {num_prompts}")

    # --------------------------------------------------
    # Extract text features
    # --------------------------------------------------

    all_features = []

    with torch.no_grad():

        for class_name in tqdm(
            class_names,
            desc="Encoding text prompts"
        ):

            prompts = templates[class_name]

            # Tokenize prompts
            tokens = clip.tokenize(prompts).to(device)

            # Encode using SenCLIP text encoder
            text_features = model.encode_text(tokens).float()

            # L2 normalization
            text_features /= text_features.norm(
                dim=-1,
                keepdim=True
            )

            all_features.append(text_features.cpu())

    # Shape: [num_classes, num_prompts, embedding_dim]
    text_features = torch.stack(all_features, dim=0)

    # --------------------------------------------------
    # Save extracted features
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "features": text_features,
            "class_names": class_names,
            "prompts": [
                templates[class_name]
                for class_name in class_names
            ],
        },
        OUTPUT_PATH
    )

    print("\nText feature extraction completed!")

    print(f"Saved to: {OUTPUT_PATH.resolve()}")
    print(f"Feature shape: {text_features.shape}")


if __name__ == "__main__":
    main()
