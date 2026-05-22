import os
import shutil
import re
import argparse

parser = argparse.ArgumentParser(description="Copy source masks using generated image filenames.")
parser.add_argument("--gen_dir", default="../../image_generation/output")
parser.add_argument("--mask_dir", default="../../image_generation/semantic_masks")
parser.add_argument("--out_dir", default="labels")
args = parser.parse_args()

gen_dir = args.gen_dir
mask_dir = args.mask_dir
out_dir = args.out_dir

os.makedirs(out_dir, exist_ok=True)

count = 0
missing = []

for name in os.listdir(gen_dir):
    if not name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # cozy_art_studio_00038.png -> 00038
    id_match = re.search(r"\d{5}", name)
    if id_match is None:
        print("Skip (no 5-digit id):", name)
        continue

    img_id = id_match.group()
    mask_name = img_id + ".png"

    src_mask = os.path.join(mask_dir, mask_name)
    dst_mask = os.path.join(out_dir, name)  # same name as generated image

    if os.path.exists(src_mask):
        shutil.copy(src_mask, dst_mask)
        count += 1
    else:
        missing.append(mask_name)

print("Copied masks:", count)
if missing:
    print("Missing masks:", set(missing))
