import torch
import clip
import torchvision
import numpy as np
import json
import random
import os
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset
from argparse import ArgumentParser, RawTextHelpFormatter


# Class names for EuroSAT dataset
NEW_CNAMES_EUROSAT = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}


def get_args():
    """
    Parses command-line arguments.
    """
    parser = ArgumentParser(description="Zero-shot classification with SenCLIP on EuroSAT dataset",
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('--num_workers', type=int, default=2, help="Number of data loading workers")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation")
    parser.add_argument('--model_arch', type=str, default='RN50', help="Model architecture to use")
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/SenCLIP_AvgPool_RN50.ckpt', help="Path to model checkpoint")
    parser.add_argument('--root_dir', type=str, default='.', help="Folder that contains Datasets/EuroSAT/2750")
    parser.add_argument('--template_path', type=str, default='prompts/prompt_eurosat_aerial_mixed.json', help="prompt_template_path")
    parser.add_argument('--max_images', type=int, default=3000,
                        help="Use a random subset of this many images (faster on CPU). 0 = use all 27,000.")
    parser.add_argument('--mean', type=float, nargs=3, default=(0.347, 0.376, 0.296), help="Normalization mean for images")
    parser.add_argument('--std', type=float, nargs=3, default=(0.269, 0.261, 0.276), help="Normalization std for images")

    return parser.parse_args()


def zeroshot_classifier(model, classnames, templates, device):
    temp_context = 'class_context' if isinstance(templates, dict) else 'unified'
    with torch.no_grad():
        zeroshot_weights = []
        for i, classname in tqdm(enumerate(classnames), desc="Encoding class prompts"):
            if temp_context == "class_context":
                texts = templates[classname] if isinstance(templates, dict) else [templates.format(classname)]
            else:
                texts = [template.format(classname) for template in templates]  # format with class

            texts = torch.cat([clip.tokenize(text) for text in texts]).to(device=device)  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device=device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Computes the top-k accuracy.
    """
    maxk = max(topk)
    pred = output.topk(maxk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum().item()) for k in topk]


def class_wise_accuracy(logits_stk, labels_stk, class_names):
    """
    Computes per-class and overall accuracy metrics.
    """
    predicted_labels = torch.argmax(logits_stk, dim=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_stk.cpu().numpy(), predicted_labels.cpu().numpy(), labels=range(len(class_names)),
        zero_division=0
    )
    class_results = {
        class_names[i]: {"Precision": round(float(precision[i]), 3),
                         "Recall": round(float(recall[i]), 3),
                         "F1-Score": round(float(f1[i]), 3)}
        for i in range(len(class_names))
    }
    return accuracy_score(labels_stk.cpu().numpy(), predicted_labels.cpu().numpy()), class_results


def transform_img(mean, std):
    """
    Returns a torchvision transform pipeline.
    """
    return Compose([
        Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean, std)
    ])


def load_model(weight_path, model_name, device):
    """
    Loads the CLIP model with specified weights.
    """
    model, _ = clip.load(model_name, device=device)
    checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=True)
    model.to(device).eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    transform = transform_img(args.mean, args.std)

    # Load dataset
    dataset_path = os.path.join(args.root_dir, 'Datasets/EuroSAT/2750/')
    dataset = ImageFolder(dataset_path, transform=transform)
    class_names = [NEW_CNAMES_EUROSAT[cls] for cls in dataset.classes]
    print(f"Class Names: {class_names}")

    eval_set = dataset
    if args.max_images and args.max_images < len(dataset):
        random.seed(0)
        idx = random.sample(range(len(dataset)), args.max_images)
        eval_set = Subset(dataset, idx)
        print(f"Using a random subset of {args.max_images} of {len(dataset)} images")
    test_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load model
    model = load_model(args.ckpt_path, args.model_arch, device)

    # Load prompts
    templates = json.load(open(args.template_path))
    encoded_prompts = zeroshot_classifier(model, class_names, templates, device).T

    logits_list, labels_list = [], []
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ encoded_prompts
            logits_list.append(logits)
            labels_list.append(labels)

    logits_stk = torch.cat(logits_list)
    labels_stk = torch.cat(labels_list)
    logits_stk = logits_stk / (logits_stk.mean(0, keepdim=True) + 1e-6)

    acc1, acc5 = accuracy(logits_stk, labels_stk, topk=(1, 5))
    overall_accuracy, class_results = class_wise_accuracy(logits_stk, labels_stk, class_names)

    print(f"Top-1 Accuracy: {acc1 / len(labels_stk):.2%}")
    print(f"Top-5 Accuracy: {acc5 / len(labels_stk):.2%}")
    print(f"Class-wise Results: {class_results}")


if __name__ == "__main__":
    args = get_args()
    main(args)
