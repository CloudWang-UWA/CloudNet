import argparse
import os, json, shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Build style-split evaluation dataset.")
    parser.add_argument("--model", choices=["Cloud", "FreestyleNet"], default="Cloud")
    parser.add_argument("--ade20k_test_dir", default="../data/ade20k/test/images")
    parser.add_argument("--generated_images_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--scene_label_file", default="../data/ade20k/test/data_ip_test.json")
    return parser.parse_args()

args = parse_args()
model = args.model
ade20k_test_dir = args.ade20k_test_dir

if args.generated_images_dir is not None:
    generated_images_dir = args.generated_images_dir
elif model == "Cloud":
    generated_images_dir = "../image_generation/output"
else:
    generated_images_dir = "../compare_models/FreestyleNet/image_generation/output"

if args.out_dir is not None:
    out_dir = args.out_dir
elif model == "Cloud":
    out_dir = "datasets_by_style"
else:
    out_dir = "datasets_by_style_free_net"

styles = ["cozy", "messy", "classic", "luxurious", "van_gogh"]

SELECTED_SCENES = [
    "bedroom",          # bedroom + hotel_room + 00045 + 00049
    "living_room",
    "bathroom",
    "kitchen",
    "dining_room",
    "conference_room",
    "corridor",
    "office",           # office + home_office
    "game_room",        # game_room + poolroom_home
    "art_studio"
]

# mkdir
os.makedirs(out_dir, exist_ok=True)
os.makedirs(f"{out_dir}/original", exist_ok=True)
for s in styles:
    os.makedirs(f"{out_dir}/{s}", exist_ok=True)

scene_label_file = args.scene_label_file

with open(scene_label_file, "r", encoding="utf-8") as f:
    scene_data = json.load(f)

selected_ids = set()

for item in scene_data:
    # image number: remove extension
    scene_number = os.path.splitext(item["image_file"])[0]

    # text -> scene_name
    text = item["text"].strip()
    if text.lower().startswith("a "):
        text = text[2:]
    elif text.lower().startswith("an "):
        text = text[3:]
    scene_name = text.replace(" ", "_").lower()

    # Merge scenes
    if (scene_name == "hotel_room" or scene_number in ["00045", "00049"]):
        scene_name = "bedroom"
    if (scene_name == "home_office"):
        scene_name = "office"
    if (scene_name == "poolroom_home"):
        scene_name = "game_room"

    if (scene_name in SELECTED_SCENES):
        selected_ids.add(scene_number)

# copy original
for scene_number in sorted(selected_ids):
    src = None
    for ext in [".jpg", ".png", ".jpeg"]:
        p = f"{ade20k_test_dir}/{scene_number}{ext}"
        if os.path.exists(p):
            src = p
            break
    if src is None:
        print("MISS original:", scene_number)
        continue
    shutil.copy2(src, f"{out_dir}/original/{os.path.basename(src)}")

# split outputs by style prefix, keep only selected_ids
for name in sorted(os.listdir(generated_images_dir)):
    if not name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    prefix = None
    for s in styles:
        if name.startswith(s + "_"):
            prefix = s
            break
    if prefix is None:
        continue

    base = os.path.splitext(name)[0]
    scene_number = base.split("_")[-1]
    if scene_number not in selected_ids:
        continue

    ext = os.path.splitext(name)[1].lower()
    shutil.copy2(f"{generated_images_dir}/{name}", f"{out_dir}/{prefix}/{name}")

print("done.")
print("original:", len(os.listdir(f"{out_dir}/original")))
for s in styles:
    print(s, ":", len(os.listdir(f"{out_dir}/{s}")))
