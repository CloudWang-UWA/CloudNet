import sys

ROOT = "../"
if ROOT not in sys.path:
    sys.path.append(ROOT)

import os
import numpy as np
import os
import json

from PIL import Image

from controlnet.tools.training_classes import get_text_label_mapping

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

SELECTED_STYLES = ["cozy", "messy", "classic", "luxurious", "van_gogh"]

STYLES_SENTENCE = [
    "a cozy room with warm lighting, wooden furniture, soft blankets, and plants",
    "a messy room with scattered objects, cluttered furniture, trash on the floor, and a chaotic environment",
    "a classic room with dark wooden furniture, ornate decorations, elegant fabrics, traditional design",
    "a luxurious room with polished wood furniture, marble surfaces, elegant lighting, modern design, and high-end materials",
    "a room in the style of van gogh with bold brush strokes, vibrant colors, thick paint texture, and post-impressionist painting style"
]

scene_label_file = "FreestyleNet/data/data_ip_test.json"
semantic_masks_dir = "FreestyleNet/data/labels"
out_dir = "FreestyleNet/image_generation/json_files"

with open(scene_label_file, "r", encoding="utf-8") as f:
    scene_data = json.load(f)

    for item in scene_data:
        # image number: remove extension
        scene_number = os.path.splitext(item["image_file"])[0]

        # text: remove leading "A " or "An ", then replace spaces with _
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
            label_map_path = f"{semantic_masks_dir}/{scene_number}.png"
            text_label_mapping = get_text_label_mapping(label_map_path)
            
            for i, style in enumerate(SELECTED_STYLES):
                text_label_mapping_style = text_label_mapping.copy()
                style_sentence = STYLES_SENTENCE[i]
                text_label_mapping_style[style_sentence] = -1

                data = {
                    "text_label_mapping": text_label_mapping_style,
                    "layout_path": label_map_path
                }

                save_path = os.path.join(out_dir, f"{style}_{scene_name}_{scene_number}.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)