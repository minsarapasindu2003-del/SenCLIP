# SenCLIP: paper-style prompt selection and DAP (our extension)

This folder holds our own implementation of the prompt-selection and classification
method described in the SenCLIP paper (alpha/beta prompt scoring + DAP), evaluated on
EuroSAT with the authors' released checkpoint. No model is trained here: we only reuse
the pretrained SenCLIP ViT-B/32 AvgPool encoders and change how prompts are selected
and scored.

The authors' own code in `src/` averages all prompt embeddings per class and its
prompt-ranking code works differently from the paper, which is why we implemented the
paper's version separately. Nothing in `src/` was modified for this folder.

## Setup

Use the same environment as the main project (Python 3.11, CPU is enough):

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/openai/CLIP.git
pip install scikit-learn tqdm
```

Files needed in the repo root (not stored in git):
- `SenCLIP_AvgPool_ViTB32.ckpt` from https://huggingface.co/pallavijainpj/SenCLIP
- `Datasets/EuroSAT/2750/<10 class folders>` (EuroSAT RGB)

Prompts come from `src/prompts/prompt_eurosat_aerial_mixed.json` (46 prompts per class).

## How to run

Run everything from this folder (`my_implementation/`) with the virtual environment active.

```
python extract_features.py        # ~30 min on CPU: encodes all 27,000 images
python extract_text_features.py   # ~20 s: encodes the 10 x 46 prompts
python check_features.py          # sanity check, should print Top-1 73.26 / Top-5 97.20
python select_prompts.py          # alpha, beta, w and best/worst K prompts per class
python evaluate_zeroshot.py       # DAP accuracy for K = 1, 2, 5, 10, 15, 25, 46
python random_baseline.py         # random-K control (30 draws per K)
```

The feature files (`features/features_*.pt`) are large and ignored by git, so each person
has to generate them once with the first two commands.

## What each script does

| File | Purpose |
|---|---|
| `extract_features.py` | Saves one L2-normalised 512-d image embedding per EuroSAT image, plus labels |
| `extract_text_features.py` | Saves normalised text embeddings, shape [10 classes, 46 prompts, 512] |
| `check_features.py` | Reproduces the authors' averaging-based result from the saved features |
| `select_prompts.py` | Computes alpha, beta and w = alpha/beta for every prompt and ranks prompts per class |
| `evaluate_zeroshot.py` | Classifies with the top-K (best) and bottom-K (worst) prompts using DAP |
| `random_baseline.py` | Same, with K random prompts per class, to test whether the ranking beats chance |
| `predict_demo.py` | Predicts the class of individual images (live demo) |

## Method

For prompt embeddings a, with C classes and T prompts per class:

- alpha(c,t) = mean similarity of prompt (c,t) to the prompts of its own class
- beta(c,t) = mean similarity of prompt (c,t) to all prompts of all classes
- w(c,t) = alpha / beta, and the K highest-w prompts are kept per class
- DAP: class = argmax over c of the product over kept prompts of
  (s . a(c,t)) / p(a(c,t)), where s is the image embedding and p(a) is the prompt's
  mean similarity over all images. The product is computed as a sum of logs.

## Results (full EuroSAT, 27,000 images, Top-1 %)

Authors' code (averaged prompt embeddings, all prompts): SenCLIP 73.26 (Top-5 97.20),
paper reports 71.22. Plain CLIP, same setup: 54.40 (Top-5 94.33),
paper reports 54.87.

DAP with the repo's positive scaling:

| K | best (by w) | worst | random mean ± std |
|---|---|---|---|
| 1 | 63.33 | 56.56 | 55.21 ± 5.35 |
| 2 | 69.97 | 53.44 | 61.57 ± 3.69 |
| 5 | 70.33 | 67.00 | 67.30 ± 2.24 |
| 10 | 70.97 | 69.57 | 69.97 ± 1.66 |
| 15 | 71.30 | 70.18 | 70.20 ± 1.44 |
| 25 | 70.78 | 70.04 | 70.53 ± 0.74 |
| 46 (all) | 71.05 | 71.05 | not applicable |

Reading: ranking prompts by w beats random choice when only a few prompts are kept
(clearest at K = 2). It does not beat using all prompts, and for K >= 10 it is within
noise of random selection.

## Differences from the paper

- Our prompt file has 46 prompts per class, the paper uses 50.
- With plain dot products, 4.72% of image-prompt similarities are zero or negative, so
  the DAP ratios are undefined. We use the scaling sim/2 + 1 from the authors' scripts
  to keep them positive. This scaling is not in the paper text. The unscaled version is
  reported as undefined in `evaluate_zeroshot.py`.
- The paper's prompt-selection figure has no printed values, so we compare trends only.
- Image preprocessing follows the authors' released code.

## Demo

```
$imgs = Get-ChildItem ..\Datasets\EuroSAT\2750 -Recurse -File | Get-Random -Count 8 | ForEach-Object FullName
python predict_demo.py $imgs --k 5
```

Prints the predicted class and the top 3 scores per image. Scores are log-ratio sums
(higher is better), not probabilities. For EuroSAT images, the folder name is used to
mark each prediction correct or wrong.