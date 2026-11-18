import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_PATH = SCRIPT_DIR / "features" / "features_text_senclip_avgpool.pt"
OUTPUT_PATH = SCRIPT_DIR / "features" / "prompt_selection_results.pt"


def main():
    print("Loading text features...")
    data = torch.load(FEATURE_PATH, map_location="cpu")

    text_features = data["features"].float()      # [C, T, D], already L2-normalised
    class_names = data["class_names"]
    prompts = data["prompts"]                     # list (class order) of lists
    C, T, D = text_features.shape
    print(f"Classes: {C} | Prompts per class: {T} | Dim: {D}")

    # K values: the paper's range, plus "all prompts" (no selection)
    top_k_values = [1, 2, 5, 10, 15, 25, T]

    # All prompt-to-prompt cosine similarities, row index = c*T + t
    all_prompts = text_features.reshape(C * T, D)
    sim = all_prompts @ all_prompts.T             # [C*T, C*T]
    sim4 = sim.reshape(C, T, C, T)

    # alpha_{c,t} = (1/T) * sum_q a_{c,t} . a_{c,q}   (same class only)
    alpha = torch.stack([sim4[c, :, c, :] for c in range(C)], dim=0).mean(dim=-1)

    # beta_{c,t} = (1/(C*T)) * sum_{d,q} a_{c,t} . a_{d,q}   (all classes)
    beta = sim.mean(dim=1).reshape(C, T)

    if torch.any(beta <= 0):
        raise ValueError("Some beta values are <= 0, check the text embeddings.")

    goodness = alpha / beta                       # w_{c,t}, shape [C, T]

    best_indices, worst_indices = {}, {}
    best_prompts, worst_prompts = {}, {}

    for k in top_k_values:
        best_indices[k], worst_indices[k] = [], []
        best_prompts[k], worst_prompts[k] = {}, {}

        for c, name in enumerate(class_names):
            order = torch.argsort(goodness[c], descending=True)
            best = order[:k].tolist()
            worst = order.flip(0)[:k].tolist()

            best_indices[k].append(best)
            worst_indices[k].append(worst)
            best_prompts[k][name] = [prompts[c][i] for i in best]
            worst_prompts[k][name] = [prompts[c][i] for i in worst]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "class_names": class_names,
        "alpha": alpha,
        "beta": beta,
        "goodness": goodness,
        "best_indices": best_indices,
        "worst_indices": worst_indices,
        "best_prompts": best_prompts,
        "worst_prompts": worst_prompts,
        "top_k_values": top_k_values,
    }, OUTPUT_PATH)

    print(f"\nSaved to: {OUTPUT_PATH.resolve()}")
    print("K values:", top_k_values)
    print("\nBest 2 prompts for", class_names[1], ":")
    for p in best_prompts[2][class_names[1]]:
        print("  -", p)


if __name__ == "__main__":
    main()