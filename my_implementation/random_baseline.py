import json
from pathlib import Path

import torch

from evaluate_zeroshot import (
    evaluate_prompts, IMAGE_FEATURE_PATH, TEXT_FEATURE_PATH, SELECTION_PATH
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "features" / "random_baseline_results.json"
N_DRAWS = 30


def main():
    image_data = torch.load(IMAGE_FEATURE_PATH, map_location="cpu")
    text_data = torch.load(TEXT_FEATURE_PATH, map_location="cpu")
    selection = torch.load(SELECTION_PATH, map_location="cpu")

    X = image_data["features"].float()
    y = image_data["labels"].long()
    T = text_data["features"].float()
    C, P, _ = T.shape

    gen = torch.Generator().manual_seed(0)
    results = {}

    print(f"{'K':>3} | {'random mean ± std':>18} | {'random range':>15} | {'best (α/β)':>10}")
    for k in [1, 2, 5, 10, 15, 25]:
        accs = []
        for _ in range(N_DRAWS):
            idx = [torch.randperm(P, generator=gen)[:k].tolist() for _ in range(C)]
            top1, _ = evaluate_prompts(X, y, T, idx, mode="repo_scale")
            accs.append(top1)
        a = torch.tensor(accs)
        best = evaluate_prompts(X, y, T, selection["best_indices"][k],
                                mode="repo_scale")[0]

        results[str(k)] = {
            "random_mean": a.mean().item(), "random_std": a.std().item(),
            "random_min": a.min().item(), "random_max": a.max().item(),
            "best": best,
        }
        print(f"{k:>3} | {a.mean():8.2f} ± {a.std():5.2f}  | "
              f"{a.min():6.2f}-{a.max():6.2f} | {best:10.2f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("\nSaved to:", OUT.resolve())


if __name__ == "__main__":
    main()