# Author: Cloud Wang
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate side-by-side CloudNet/FreestyleNet comparison images.")
parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
parser.add_argument("--input_dir", default="datasets_by_style")
parser.add_argument("--freestylenet_dir", default="datasets_by_style_free_net")
parser.add_argument("--semantic_masks_dir", default="../image_generation/semantic_masks")
parser.add_argument("--clip_result_file", default="CLIP_score/clip_per_image_result.txt")
parser.add_argument("--clip_result_file_free_net", default="CLIP_score/clip_per_image_result_free_net.txt")
parser.add_argument("--clip_aes_file", default="CLIP_aes/clip_aesthetic_all_images.txt")
parser.add_argument("--clip_aes_file_free_net", default="CLIP_aes/clip_aesthetic_all_images_free_net.txt")
parser.add_argument("--vis_dir", default="comparison/mine_vs_free_net")
parser.add_argument("--raw_images_dir", default="comparison/raw_images")
parser.add_argument("--raw_only", action="store_true", help="Only export raw images; skip side-by-side comparison grids.")
parser.add_argument("--no_metrics", action="store_true", help="Do not draw CLIP/CLIP-AES metric panels on generated images.")
args = parser.parse_args()

ROOT = args.project_root
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import os

from PIL import Image, ImageDraw, ImageFont

from controlnet.tools.training_classes import (
    get_class_stacks, 
    make_one_hot,
    get_cs_classes,
    get_cs_palette,
    map_label2RGB,
    remap_label_map
)

from image_generation import (
    add_title,
    make_legend,
    concat_one_line
)

styles = ["cozy", "messy", "classic", "luxurious", "van_gogh"]

input_dir = args.input_dir
input_dir_free_net_dir = args.freestylenet_dir
semantic_masks_dir = args.semantic_masks_dir
clip_result_file = args.clip_result_file
clip_result_file_free_net = args.clip_result_file_free_net
clip_aes_file = args.clip_aes_file
clip_aes_file_free_net = args.clip_aes_file_free_net
vis_dir = args.vis_dir
raw_images_dir = args.raw_images_dir
os.makedirs(os.path.join(raw_images_dir, "my_model"), exist_ok=True)
os.makedirs(os.path.join(raw_images_dir, "free_net"), exist_ok=True)

def load_clip_scores(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name, score = line.strip().split("\t")
            base = os.path.splitext(name)[0]

            scene_number = base.split("_")[-1]

            style = None
            for s in styles:
                if base.startswith(s + "_"):
                    style = s
                    break
            if style is None:
                continue

            d[(scene_number, style)] = float(score)
    return d

def make_metric_panel(height, lines, width=160, font_size=14):
    panel = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(panel)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw.multiline_text((10, 10), "\n".join(lines), fill=(0,0,0), font=font, spacing=6)
    draw.rectangle([0,0,width-1,height-1], outline=(200,200,200))
    return panel

def draw_metric_on_image(img, lines, bg=(255,255,255)):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    W, H = img.size

    # scale with image size
    ref = min(W, H)
    font_size = int(ref * 0.028)
    font_size = max(10, min(14, font_size))

    pad = max(4, int(font_size * 0.45))
    margin = max(6, int(font_size * 0.60))

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # measure text box
    line_h = font.getbbox("Ag")[3]
    max_w = 0
    for l in lines:
        w = draw.textlength(l, font=font)
        max_w = max(max_w, w)

    box_w = int(max_w + pad*2)
    box_h = int(line_h * len(lines) + pad*2 + 4)

    x2 = W - margin
    y2 = H - margin
    x1 = x2 - box_w
    y1 = y2 - box_h

    # background box
    draw.rectangle([x1, y1, x2, y2], fill=bg, outline=(180,180,180))

    # text
    y = y1 + pad
    for l in lines:
        draw.text((x1 + pad, y), l, fill=(0,0,0), font=font)
        y += line_h + 2

    return img

def generate_vis(original_image_name, model_type = "my_model", export_raw=False):
    # Get the scene number from the original image file name
    scene_number = os.path.splitext(original_image_name)[0]
    scene_name = ""
    original_image = None
    in_dir = ""
    if model_type == "my_model":
        in_dir = input_dir
    elif model_type == "free_net":
        in_dir = input_dir_free_net_dir
    else:
        raise ValueError("Unknown model type")

    # Style images
    images  = []
    original_image = Image.open(f"{in_dir}/original/{original_image_name}")
    for style in styles:
        style_dir = f"{in_dir}/{style}"
        for s_image in os.listdir(style_dir):
            if scene_number in s_image:
                images.append(Image.open(f"{style_dir}/{s_image}"))

                base = os.path.splitext(s_image)[0]          # classic_art_studio_00038
                scene_name = base[len(style)+1:].rsplit("_", 1)[0]  # art_studio
                break

    # Get the label map
    label_map = Image.open(f"{semantic_masks_dir}/{scene_number}.png")
    label_map_train = remap_label_map(label_map)
    cs_classes = get_cs_classes()
    cs_palette = get_cs_palette()
    legend_img = make_legend(label_map_train, cs_classes, cs_palette, target_height=label_map.height)
    label_map_rgb = Image.fromarray(map_label2RGB(np.array(label_map)))

    # export raw images
    if export_raw:
        scene_dir = os.path.join(raw_images_dir, model_type, scene_number)
        os.makedirs(scene_dir, exist_ok=True)

        original_image.save(os.path.join(scene_dir, "original.png"))
        label_map_rgb.save(os.path.join(scene_dir, "mask.png"))
        legend_img.save(os.path.join(scene_dir, "legend.png"))

        for style, img in zip(styles, images):
            img.save(os.path.join(scene_dir, f"{style}.png"))

    if args.raw_only:
        return None

    titled_images = [
        add_title(original_image, "Original Image"),
        add_title(label_map_rgb, f"Semantic Mask ({scene_name}_{scene_number})"),
        add_title(legend_img, "Legend")
    ]

    clip_scores = clip_scores_my if model_type=="my_model" else clip_scores_free
    clip_aes = clip_aes_my if model_type=="my_model" else clip_aes_free

    for style, img in zip(styles, images):
        clip_s = clip_scores.get((scene_number, style), None)
        aes_s  = clip_aes.get((scene_number, style), None)

        lines = []
        lines.append("CLIP-score: N/A" if clip_s is None else f"CLIP-score: {clip_s:5.2f}")
        lines.append("CLIP-aes:  N/A" if aes_s  is None else f"CLIP-aes:  {aes_s:5.2f}")

        img_with_metric = img if args.no_metrics else draw_metric_on_image(img, lines)

        titled_images.append(add_title(img_with_metric, f"Generated ({style})"))

    grid = concat_one_line(titled_images, save_path = None)
    return grid

def concat_vertical(img_top, img_bottom, gap=20, bg=(255,255,255)):
    w = max(img_top.width, img_bottom.width)
    h = img_top.height + img_bottom.height + gap

    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(img_top, (0, 0))
    canvas.paste(img_bottom, (0, img_top.height + gap))
    return canvas

os.makedirs(vis_dir, exist_ok=True)
original_images = sorted(os.listdir(f"{input_dir}/original"))
original_images_free_net = sorted(os.listdir(f"{input_dir_free_net_dir}/original"))
if args.no_metrics or args.raw_only:
    clip_scores_my = {}
    clip_scores_free = {}
    clip_aes_my = {}
    clip_aes_free = {}
else:
    clip_scores_my = load_clip_scores(clip_result_file)
    clip_scores_free = load_clip_scores(clip_result_file_free_net)
    clip_aes_my = load_clip_scores(clip_aes_file)
    clip_aes_free = load_clip_scores(clip_aes_file_free_net)

for i, fname in enumerate(original_images):
    my_vis = generate_vis(fname, model_type="my_model", export_raw=True)
    free_net_vis = generate_vis(fname, model_type="free_net", export_raw=True)

    if args.raw_only:
        continue

    both = concat_vertical(my_vis, free_net_vis, gap=30)
    both.save(f"{vis_dir}/{os.path.splitext(fname)[0]}_compare.png")
