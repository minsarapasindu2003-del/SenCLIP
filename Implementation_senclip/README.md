# SenCLIP: local implementation (Image Processing and Computer Vision module)

Based on the paper *SenCLIP: Enhancing zero-shot land-use mapping for Sentinel-2 with ground-level prompting* (WACV 2025)
and the official repository https://github.com/pallavijain-pj/SenCLIP.

## What is in this folder
- **Own training (Google Colab):** SenCLIP RN50 with average pooling, trained on a subset of 8,000 Sentinel-2 locations
  for 20 epochs (the paper uses all ~235k locations). Frozen LUCAS ground-photo embeddings from the authors' Hugging Face repo.
- `zeroshot.py`: zero-shot evaluation on EuroSAT (official script, with a `--max_images` option for CPU laptops).
- `eval_all.py`: evaluates CLIP, the official SenCLIP checkpoints and my own model on EuroSAT with aerial, ground and mixed prompts.
- `senclip_demo.py` / `senclip_demo.ipynb`: prompt selection and classification for my own images and classes.
  Prompts are ranked per model with the paper's score (Eq. 6-7): **w = alpha / beta**,
  where alpha is the mean similarity of a prompt to the prompts of its own class and beta its mean similarity to all prompts.
  The top-K prompts of each class (ground view and aerial view) are used to classify each image.
- `prompts/`: LLM-generated prompts (the authors' EuroSAT / BigEarthNet prompts and my own classes: Harbour, Air Port, Paddy Field).
- `my_image/`: my test images (airport, harbour, paddy field).
- `setup_data.py`, `download_all_ckpt.py`: download the checkpoints and EuroSAT.

## Not included (too large for GitHub)
`venv/`, `checkpoints/` (model weights), `Datasets/` (EuroSAT). Recreate them as below.

## How to run
```
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow tqdm scikit-learn matplotlib ftfy regex transformers huggingface_hub ipykernel
pip install git+https://github.com/openai/CLIP.git

python setup_data.py            # EuroSAT + CLIP baseline + RN50 checkpoints
python download_all_ckpt.py     # all official checkpoints (RN50 and ViT-B/32)

python eval_all.py --max_images 1000
python senclip_demo.py --image_dir my_image --ground_json prompts/my_ground.json --aerial_json prompts/my_aerial.json
```
My own trained checkpoint (`last.ckpt`) is not in the repo because of its size; put it in `checkpoints/` to include it in the comparison.

## Observations
- On 1,000 EuroSAT images, plain CLIP RN50 gave 48.5% (aerial prompts) and 28.9% (ground prompts), close to the paper.
  The official RN50 SenCLIP checkpoints scored 12 to 20% with the released zero-shot script, well below the paper. The cause was not found.
- On my 6 own images, the official ViT-B/32 models were best (5/6 to 6/6); RN50 models were weaker (2/6 to 3/6).
  Prompt rankings are identical for models that share a backbone, because training only changes the image encoder.
- Only 6 images were used, so these numbers are a rough indication.
