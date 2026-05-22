import os
import json
import shutil

model = "FreestyleNet"
ade20k_test_dir = "../data/ade20k/test/images"
generated_images_dir = ""
out_dir = ""

if model == "Cloud":
    generated_images_dir  = "../image_generation/output"
    out_dir = "datasets_by_scene"
elif model == "FreestyleNet":
    generated_images_dir  = "../compare_models/FreestyleNet/image_generation/output"
    out_dir = "datasets_by_scene_free_net"

SELECTED_SCENES = [
    "bedroom",
    "living_room",
    "bathroom",
    "kitchen",
    "dining_room",
    "conference_room",
    "corridor",
    "office",
    "game_room",
    "art_studio"
]

styles = ["cozy", "messy", "classic", "luxurious", "van_gogh"]

# Create folders
os.makedirs(out_dir, exist_ok=True)
for s in SELECTED_SCENES:
    os.makedirs(f"{out_dir}/{s}", exist_ok=True)

scene_label_file = "../data/ade20k/test/data_ip_test.json"
with open(scene_label_file, "r", encoding="utf-8") as f:
    scene_data = json.load(f)

# Map image numbers to scene names
image_to_scene = {}
for item in scene_data:
    scene_number = os.path.splitext(item["image_file"])[0]
    text = item["text"].strip()
    if text.lower().startswith("a "):
        text = text[2:]
    elif text.lower().startswith("an "):
        text = text[3:]
    scene_name = text.replace(" ", "_").lower()

    # Merge scenes
    if scene_name == "hotel_room" or scene_number in ["00045", "00049"]:
        scene_name = "bedroom"
    if scene_name == "home_office":
        scene_name = "office"
    if scene_name == "poolroom_home":
        scene_name = "game_room"

    if scene_name in SELECTED_SCENES:
        image_to_scene[scene_number] = scene_name

# Counter for each scene/style
scene_style_counter = {}

# Copy generated images to scene/style folders
for filename in sorted(os.listdir(generated_images_dir)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    name = os.path.splitext(filename)[0]

    # -------- Detect style --------
    style = None
    for s in styles:
        if name.startswith(s + "_"):
            style = s
            break

    if style is None:
        print(f"Style not found in filename {filename}")
        continue

    # Remove style + "_"
    rest = name[len(style) + 1:]   # e.g. art_studio_00038

    parts = rest.split("_")
    if len(parts) < 2:
        continue

    scene_number = parts[-1]
    scene_from_filename = "_".join(parts[:-1]).lower()

    # -------- Validate scene --------
    scene_name = image_to_scene.get(scene_number)
    if scene_name is None:
        continue

    if scene_name != scene_from_filename:
        continue

    # -------- Create scene/style folder --------
    scene_style_dir = os.path.join(out_dir, scene_name, style)
    os.makedirs(scene_style_dir, exist_ok=True)

    src_path = os.path.join(generated_images_dir, filename)
    dst_path = os.path.join(scene_style_dir, filename)

    shutil.copy2(src_path, dst_path)

    key = f"{scene_name}/{style}"
    scene_style_counter[key] = scene_style_counter.get(key, 0) + 1


# -------- Write summary --------
summary_file = os.path.join(out_dir, "scene_style_counts.txt")
with open(summary_file, "w") as f:
    for key, count in sorted(scene_style_counter.items()):
        f.write(f"{key}: {count}\n")

print("Done! Images sorted by scene/style.")
print(f"Summary saved to {summary_file}")