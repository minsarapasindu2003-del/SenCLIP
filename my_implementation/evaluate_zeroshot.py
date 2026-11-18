import json
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_FEATURE_PATH = SCRIPT_DIR / "features" / "features_senclip_avgpool.pt"
TEXT_FEATURE_PATH = SCRIPT_DIR / "features" / "features_text_senclip_avgpool.pt"
SELECTION_PATH = SCRIPT_DIR / "features" / "prompt_selection_results.pt"
OUTPUT_PATH = SCRIPT_DIR / "features" / "evaluation_results.json"

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


def evaluate_prompts(image_features, labels, text_features,
                     selected_indices, mode="paper"):
    """
    DAP: c_i = argmax_c prod_t (s_i . a_{c,t}) / p(a_{c,t}),
    computed as a sum of logs (same argmax).

    mode="paper":       plain dot products. Returns None if any similarity
                        is <= 0 (the ratio is undefined).
    mode="repo_scale":  sim/2 + 1 (the repo's scaling, not in the paper text).
    """
    num_classes = text_features.shape[0]
    class_scores = []

    for c in range(num_classes):
        selected_text = text_features[c, selected_indices[c]]   # [K, D]
        sim = image_features @ selected_text.T                  # [N, K]

        if mode == "repo_scale":
            sim = sim / 2.0 + 1.0

        if not torch.all(sim > 0):
            return None

        prior = sim.mean(dim=0)                                 # p(a) over all images
        class_scores.append(torch.log(sim / prior).sum(dim=1))

    logits = torch.stack(class_scores, dim=1)                   # [N, C]

    top1 = (logits.argmax(dim=1) == labels).float().mean().item() * 100
    top5_idx = logits.topk(5, dim=1).indices
    top5 = (top5_idx == labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    return top1, top5


def main():
    print("Loading saved features...")
    image_data = torch.load(IMAGE_FEATURE_PATH, map_location="cpu")
    text_data = torch.load(TEXT_FEATURE_PATH, map_location="cpu")
    selection = torch.load(SELECTION_PATH, map_location="cpu")

    image_features = image_data["features"].float()
    labels = image_data["labels"].long()
    text_features = text_data["features"].float()
    class_names = text_data["class_names"]

    expected = [CLASS_NAME_MAPPING[f] for f in image_data["class_folders"]]
    if class_names != expected:
        raise ValueError("Image and text feature class orders do not match.")

    print(f"Images: {len(image_features)} | Classes: {len(class_names)}")

    # How widespread are non-positive similarities?
    all_text = text_features.reshape(-1, text_features.shape[-1])
    all_sim = image_features @ all_text.T
    print(f"Image-prompt cosine: min {all_sim.min():.3f} | "
          f"mean {all_sim.mean():.3f} | max {all_sim.max():.3f}")
    print(f"Non-positive values: {(all_sim <= 0).float().mean().item() * 100:.2f}%")
    del all_sim

    results = {}

    for mode in ["paper", "repo_scale"]:
        results[mode] = {}
        print(f"\n===== DAP mode: {mode} =====")

        for k in selection["top_k_values"]:
            results[mode][str(k)] = {}

            for kind in ["best", "worst"]:
                idx = selection[f"{kind}_indices"][k]
                out = evaluate_prompts(image_features, labels, text_features,
                                       idx, mode=mode)
                if out is None:
                    results[mode][str(k)][kind] = None
                    print(f"K={k:>2} {kind:<5} | undefined (non-positive similarities)")
                    continue

                top1, top5 = out
                results[mode][str(k)][kind] = {"top1_accuracy": top1,
                                               "top5_accuracy": top5}
                print(f"K={k:>2} {kind:<5} | Top-1: {top1:6.2f}% | Top-5: {top5:6.2f}%")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()