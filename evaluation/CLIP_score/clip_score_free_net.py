import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.append(ROOT)

import os
import numpy as np
from PIL import Image
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import time

from compare_models.FreestyleNet.configs.free_net_style_map import STYLE_SENTENCE_TO_TAG

t0 = time.time()

base_dir = "../datasets_by_style_free_net"
styles = ["cozy", "messy", "classic", "luxurious", "van_gogh"]

prompts_file = "../../compare_models/FreestyleNet/image_generation/prompts/prompts.txt"

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images).astype("uint8")
    with torch.no_grad():
        clip_score = clip_score_fn(
            torch.from_numpy(images_int).permute(0, 3, 1, 2), 
            prompts
        )
    return round(float(clip_score), 4)

id2prompt = {}

with open(prompts_file, "r", encoding="utf-8") as f:
    for line in f:
        img_id, full_prompt = line.strip().split(maxsplit=1)

        style_tag = None
        for style_sentence, tag in STYLE_SENTENCE_TO_TAG.items():
            if style_sentence in full_prompt:
                style_tag = tag
                break

        if style_tag is None:
            raise ValueError(f"No style matched in prompt: {full_prompt}")

        new_id = f"{style_tag}_{img_id}"   # luxurious_00605

        if new_id not in id2prompt:
            id2prompt[new_id] = full_prompt

# ===== compute scores =====
all_scores = []
style_scores = {s: [] for s in styles}
all_results = []   # (style, name, score)

for style in styles:
    image_dir = os.path.join(base_dir, style)

    for name in sorted(os.listdir(image_dir)):
        if name.endswith(".jpg"):
            base = os.path.splitext(name)[0]
            img_num = base.split("_")[-1]

            img_id = f"{style}_{img_num}"          # luxurious_00605
            prompt = id2prompt[img_id]

            image = Image.open(os.path.join(image_dir, name)) # eg: shape (W, H): (500, 375) 
            image_np = np.array(image) # eg: shape (H, W, C): (375, 500, 3)
            image_np_4d = np.expand_dims(image_np, axis=0) # shape (1, H, W, C): (1, 375, 500, 3)

            score = calculate_clip_score(image_np_4d, [prompt])

            all_scores.append(score)
            style_scores[style].append(score)
            all_results.append((name, score))

# ===== overall =====
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

# ===== save averages (ONE file) =====
with open("clip_avg_result_free_net.txt", "w", encoding="utf-8") as f:
    f.write("Overall_CLIP_score\tNumImages\tTime(seconds)\n")
    f.write(f"{mean_score}\t{len(all_scores)}\t{total_time}\n\n")

    f.write("Style\tNumImages\tCLIPScore\n")
    for s in styles:
        if style_means[s] is None:
            f.write(f"{s}\t0\tNA\n")
        else:
            f.write(f"{s}\t{len(style_scores[s])}\t{style_means[s]}\n")

# ===== save per-image (ONE file for all styles) =====
with open("clip_per_image_result_free_net.txt", "w", encoding="utf-8") as f:
    for name, score in all_results:
        f.write(f"{name}\t{score}\n")
