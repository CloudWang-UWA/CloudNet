import os
import argparse
import torch
import clip
from PIL import Image
from dl_clip_aes_model import get_aesthetic_model   # your function

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Calculate CLIP aesthetic score.")
parser.add_argument("--model", choices=["Cloud", "FreestyleNet"], default="Cloud")
parser.add_argument("--base_dir", default=None)
parser.add_argument("--output_dir", default=".")
parser.add_argument("--clip_dir", default="../../clip/models")
args = parser.parse_args()

CLIP_DIR = "../../clip/models"
CLIP_DIR = args.clip_dir
styles = ["cozy", "messy", "classic", "luxurious", "van_gogh"]

model = args.model

base_dir = ""
per_image_file = ""
avg_file = ""

if args.base_dir is not None:
    base_dir = args.base_dir
    per_image_file = "clip_aesthetic_all_images.txt"
    avg_file = "clip_aesthetic_avg_all.txt"
elif model == "Cloud":
    base_dir = "../datasets_by_style"
    per_image_file = "clip_aesthetic_all_images.txt"
    avg_file = "clip_aesthetic_avg_all.txt"
elif model == "FreestyleNet":
    base_dir = "../datasets_by_style_free_net"
    per_image_file = "clip_aesthetic_all_images_free_net.txt"
    avg_file = "clip_aesthetic_avg_all_free_net.txt"

os.makedirs(args.output_dir, exist_ok=True)

# 1. load CLIP
clip_model, preprocess = clip.load(
    "ViT-L/14",
    device=device,
    download_root=CLIP_DIR
)

clip_model.eval()

# 2. load aesthetic head
aes_model = get_aesthetic_model("vit_l_14").to(device).eval()

@torch.no_grad()
def aesthetic_score_image_folder(folder):
    scores = []
    names = []

    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img = Image.open(os.path.join(folder, name)).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)

        feat = clip_model.encode_image(img).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)   # IMPORTANT

        score = aes_model(feat).item()
        scores.append(score)
        names.append(name)

    return names, scores

# ===== styles evaluation (one per-image file, one avg file) =====
all_image_records = []   # (style, name, score)
all_scores = []
style_stats = []         # (style, num, avg)

for style in styles:
    image_dir = os.path.join(base_dir, style)
    names, scores = aesthetic_score_image_folder(image_dir)

    style_avg = sum(scores) / len(scores)
    style_stats.append((style, len(scores), style_avg))

    for n, s in zip(names, scores):
        all_image_records.append((style, n, s))
        all_scores.append(s)

    print(f"{style} mean aesthetic score: {style_avg}")

# overall average
overall_avg = sum(all_scores) / len(all_scores)
print("Overall mean aesthetic score:", overall_avg)

# ===== save per-image results (ALL styles) =====
per_image_path = os.path.join(args.output_dir, per_image_file)
avg_path = os.path.join(args.output_dir, avg_file)

with open(per_image_path, "w", encoding="utf-8") as f:
    for style, name, score in all_image_records:
        f.write(f"{name}\t{score}\n")

# ===== save averages =====
with open(avg_path, "w", encoding="utf-8") as f:
    f.write("Style\tNumImages\tAverageAestheticScore\n")
    for style, num, avg in style_stats:
        f.write(f"{style}\t{num}\t{avg}\n")
    f.write(f"\nOverall\t{len(all_scores)}\t{overall_avg}\n")

print("Saved per-image results to:", per_image_path)
print("Saved averages to:", avg_path)
