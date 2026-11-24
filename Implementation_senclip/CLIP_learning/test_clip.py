import torch
import clip
from PIL import Image

# 1. Choose the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 2. Load the pretrained CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# 3. Image filenames
image_files = [
    "CLIP_learning/airport_1.jpg",
    "CLIP_learning/airport_2.jpg",
    "CLIP_learning/airport_3.jpg"
]

# 4. Candidate descriptions
descriptions = [
    "An airport",
    "A highway",
    "A building complex",
    "A park",
    "A satellite image"
]

# 5. Convert descriptions into tokens
text_tokens = clip.tokenize(descriptions).to(device)

# 6. Classify each image
for filename in image_files:

    print("\nImage:", filename)

    # Open and prepare the image
    image = Image.open(filename).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Compare the image with the descriptions
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        # Normalize the feature vectors
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity scores
        similarity = (image_features @ text_features.T).squeeze(0)

        # Convert scores into relative percentages
        probabilities = similarity.softmax(dim=0)

    # Print results
    for description, probability in zip(descriptions, probabilities):
        print(f"{description}: {probability.item() * 100:.2f}%")

    best_index = probabilities.argmax().item()
    print("Best matching description:", descriptions[best_index])