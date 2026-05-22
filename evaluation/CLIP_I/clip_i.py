import os
import argparse
import numpy as np
from PIL import Image
import torch
import time
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model.eval()

t0 = time.time()

parser = argparse.ArgumentParser(description="Calculate CLIP image-image style similarity.")
parser.add_argument("--base_dir", default="../datasets_by_style")
parser.add_argument("--style_dir", default="../../image_generation/style_images")
parser.add_argument("--output_dir", default=".")
args = parser.parse_args()

base_dir = args.base_dir
style_dir = args.style_dir
os.makedirs(args.output_dir, exist_ok=True)
styles = ["cozy", "messy", "classic", "luxurious", "van_gogh"]

def calculate_clip_score(image1, image2):
    inputs = processor(images=[image1, image2], return_tensors="pt")

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # Normalize
    image_features = F.normalize(image_features, dim=-1)

    # Cosine similarity
    similarity = (image_features[0] @ image_features[1].T).item()

    return round(similarity, 4)

# ===== collect results =====
all_scores = []
all_results = []              # (style, name, score)
style_scores = {s: [] for s in styles}

for style in styles:
    image_dir = os.path.join(base_dir, style)

    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".png"):
            # Example: classic_art_studio_00038.png
            name = filename.replace(".png", "")
            parts = name.split("_")

            # Remove style + underscore from start to get remaining, art_studio_00038
            rest = name[len(style) + 1:]

            # [art, studio]
            scene_parts = rest.split("_")[:-1]

            # art_studio
            scene = "_".join(scene_parts) 

            # Construct reference image path
            ref_image_name = f"{style}_{scene}.jpg"
            ref_path = os.path.join(style_dir, scene, ref_image_name)

            image = Image.open(os.path.join(image_dir, filename)).convert("RGB") # eg: shape (W, H): (500, 375) 
            style_image = Image.open(ref_path).convert("RGB") # eg: shape (W, H): (500, 375)

            score = calculate_clip_score(image, style_image)

            all_scores.append(score)
            style_scores[style].append(score)
            all_results.append((filename, score))

mean_score = sum(all_scores) / len(all_scores)
print("Overall CLIP score:", mean_score)

t1 = time.time()
total_time = t1 - t0
print("Time:", total_time, "seconds")

# ===== per-style means =====
style_means = {}
for s in styles:
    style_means[s] = (sum(style_scores[s]) / len(style_scores[s])) if len(style_scores[s]) > 0 else None
    print(f"{s} CLIP score:", style_means[s])

# ===== save averages (ONE file, same idea as aes) =====
with open(os.path.join(args.output_dir, "clip_avg_result.txt"), "w", encoding="utf-8") as f:
    f.write("Overall_CLIP_I_score\tNumImages\tTime(seconds)\n")
    f.write(f"{mean_score}\t{len(all_scores)}\t{total_time}\n\n")

    f.write("Style\tNumImages\tCLIP_I_Score\n")
    for s in styles:
        if style_means[s] is None:
            f.write(f"{s}\t0\tNA\n")
        else:
            f.write(f"{s}\t{len(style_scores[s])}\t{style_means[s]}\n")

# ===== save per-image (ONE file for all styles) =====
with open(os.path.join(args.output_dir, "clip_per_image_result.txt"), "w", encoding="utf-8") as f:
    f.write("Image\tCLIP_I_Score\n")
    for filename, score in all_results:
        f.write(f"{filename}\t{score}\n")
