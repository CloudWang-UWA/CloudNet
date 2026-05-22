# Author: Cloud Wang

import numpy as np
import os
import json
import argparse

import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from ip_adapter import IPAdapter, IPAdapterPlus
except ModuleNotFoundError:
    from train_ip_adapter.IP_Adapter.ip_adapter.ip_adapter import IPAdapter, IPAdapterPlus

from controlnet.tools.training_classes import (
    get_class_stacks, 
    make_one_hot,
    get_cs_classes,
    get_cs_palette,
    map_label2RGB,
    remap_label_map
)

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

# styles: ["cozy", "messy", "classic", "luxurious", "van_gogh"]

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Style controllable indoor image generation")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="image_generation/output",
        help="output image save path",
    )
    parser.add_argument(
        "--prompt_folder",
        type=str,
        default="image_generation/prompts",
        help="prompt text file path",
    )
    parser.add_argument(
        "--vis_folder",
        type=str,
        default="image_generation/visualization",
        help="visualization save path",
    )
    parser.add_argument(
        "--original_folder",
        type=str,
        default="image_generation/original_images",
        help="original images path",
    )
    parser.add_argument(
        "--semantic_masks_folder",
        type=str,
        default="image_generation/semantic_masks",
        help="semantic masks path",
    )
    parser.add_argument(
        "--style_images_folder",
        type=str,
        default="image_generation/style_images",
        help="style images path",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="base model path",
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        default="cwang_model/controlnet_97_classes",
        help="ControlNet model path",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="VAE model path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="ip_adapter_models/models/image_encoder",
        help="image encoder model path",
    )
    parser.add_argument(
        "--ip_adapter_path",
        type=str,
        default="ip_adapter_models/models/ip-adapter-plus_sd15.bin",
        help="IP-Adapter model path",
    )
    parser.add_argument(
        "--ip_adapter_type",
        choices=["normal", "plus"],
        default="plus",
        help="IP-Adapter variant to use.",
    )
    parser.add_argument(
        "--scene_label_file",
        type=str,
        default="data/ade20k/test/data_ip_test.json",
        help="scene type file path",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--num_generated_images",
        type=int,
        default=1,
        help="Number of generated images per prompt.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Strength of image prompt (IP-Adapter).",
    )
    parser.add_argument(
        "--styles",
        type=list, 
        default=["cozy", "messy", "classic", "luxurious", "van_gogh"],
        help="Indoor scene styles"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def pad_label_mask_to_8(label_mask, ignore_value=0):
    """
    return:
        padded_mask (PIL Image),
        (orig_w, orig_h)
    """
    if isinstance(label_mask, np.ndarray):
        label_mask = Image.fromarray(label_mask.astype(np.uint8))

    w, h = label_mask.size

    new_w = (w + 7) // 8 * 8
    new_h = (h + 7) // 8 * 8

    pad_w = new_w - w
    pad_h = new_h - h

    padded = ImageOps.expand(
        label_mask,
        border=(0, 0, pad_w, pad_h),  # left, top, right, bottom
        fill=ignore_value
    )

    return padded, (w, h)

def add_title(img, title, bar_h=36, bg=(255,255,255), fg=(0,0,0)):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    w, h = img.size
    out = Image.new("RGB", (w, h + bar_h), bg)
    out.paste(img, (0, bar_h))

    draw = ImageDraw.Draw(out)
    text_w = draw.textlength(title, font=font)
    draw.text(((w - text_w) // 2, (bar_h - 16) // 2),
              title, fill=fg, font=font)

    return out

def make_legend(label_map_train, cs_classes, cs_palette,
                target_height, box=18, pad=10, col_gap=30):
    label_map_train = np.array(label_map_train)
    cs_palette = np.array(cs_palette, dtype=np.uint8)

    ids, counts = np.unique(label_map_train, return_counts=True)
    total = label_map_train.size

    items = []
    for cid, cnt in zip(ids, counts):
        cid = int(cid)
        if cid < 0 or cid >= len(cs_palette):
            continue
        name = cs_classes[cid] if isinstance(cs_classes, list) else cs_classes.get(cid, str(cid))
        items.append((cid, name, cnt / total))

    # sort by area
    items.sort(key=lambda x: x[2], reverse=True)

    # font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    line_h = max(box, 18) + 6
    rows_per_col = max(1, (target_height - 2 * pad) // line_h)
    cols = int(np.ceil(len(items) / rows_per_col))

    # measure max text width
    dummy = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(dummy)
    max_text_w = max(
        d.textlength(f"{name} ({pct*100:.1f}%)", font=font)
        for _, name, pct in items
    )

    col_w = pad + box + 8 + int(max_text_w) + pad
    width = cols * col_w + (cols - 1) * col_gap

    legend = Image.new("RGB", (width, target_height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)

    for i, (cid, name, pct) in enumerate(items):
        col = i // rows_per_col
        row = i % rows_per_col

        x0 = col * (col_w + col_gap) + pad
        y0 = pad + row * line_h

        rgb = tuple(int(v) for v in cs_palette[cid])
        draw.rectangle([x0, y0, x0 + box, y0 + box], fill=rgb, outline=(0, 0, 0))
        draw.text((x0 + box + 8, y0 - 1),
                  f"{name} ({pct*100:.1f}%)",
                  fill=(0, 0, 0),
                  font=font)

    return legend

def concat_one_line(images, save_path=None):
    h0 = images[0].size[1]
    assert all(im.size[1] == h0 for im in images), "heights not equal"

    total_w = sum(im.size[0] for im in images)
    grid = Image.new("RGB", (total_w, h0))

    x = 0
    for im in images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        grid.paste(im, (x, 0))
        x += im.size[0]

    if save_path:
        grid.save(save_path)
    return grid

def main(args):
    # directories path
    output_dir = args.output_folder
    os.makedirs(output_dir, exist_ok=True)
    prompt_dir = args.prompt_folder
    os.makedirs(prompt_dir, exist_ok=True)
    vis_dir = args.vis_folder
    os.makedirs(vis_dir, exist_ok=True)
    original_dir = args.original_folder
    os.makedirs(original_dir, exist_ok=True)
    semantic_masks_dir = args.semantic_masks_folder
    os.makedirs(semantic_masks_dir, exist_ok=True)
    style_images_dir = args.style_images_folder
    os.makedirs(style_images_dir, exist_ok=True)

    # models path
    base_model_path = args.base_model_path
    controlnet_model_path = args.controlnet_model_path
    vae_model_path = args.vae_model_path
    image_encoder_path = args.image_encoder_path
    ip_adapter_path = args.ip_adapter_path

    scene_label_file = args.scene_label_file
    styles = args.styles
    num_inference_steps = args.num_inference_steps
    num_generated_images = args.num_generated_images
    scale = args.scale

    prompt_file = os.path.join(prompt_dir, "prompts.txt")

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    # load vae
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    # load controlnet
    controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

    # load SD pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    # load ip-adapter
    if args.ip_adapter_type == "normal":
        ip_model = IPAdapter(pipe, image_encoder_path, ip_adapter_path, "cuda")
    else:
        ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_adapter_path, "cuda", num_tokens=16)

    with open(scene_label_file, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    with open(prompt_file, "w", encoding="utf-8") as pf:
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
                # read image prompt
                label_map = Image.open(f"{semantic_masks_dir}/{scene_number}.png")
                original_image = Image.open(f"{original_dir}/{scene_number}.jpg")

                label_map_pad, (orig_w, orig_h) = pad_label_mask_to_8(label_map, ignore_value=0)

                label_map_one_hot = torch.Tensor(make_one_hot(label_map_pad))
                label_map_one_hot = torch.unsqueeze(label_map_one_hot.permute(2, 0, 1), 0)
                prompt = get_class_stacks(label_map)

                label_map_train = remap_label_map(label_map)
                cs_classes = get_cs_classes()
                cs_palette = get_cs_palette()
                legend_img = make_legend(label_map_train, cs_classes, cs_palette, target_height=label_map.height)

                images = []
                for style in styles:
                    style_image = Image.open(f"{style_images_dir}/{scene_name}/{style}_{scene_name}.jpg")

                    pf.write(f"{scene_number} {prompt}\n")

                    # generate image variations
                    generated_image = ip_model.generate(
                        pil_image=style_image, 
                        image=label_map_one_hot, 
                        prompt=prompt, 
                        num_samples=num_generated_images, 
                        num_inference_steps=num_inference_steps,
                        scale=scale,
                        seed=42
                    )

                    current_image = generated_image[0]
                    current_image = current_image.crop((0, 0, orig_w, orig_h))
                    images.append(current_image)

                    current_image.save(f"{output_dir}/{style}_{scene_name}_{scene_number}.png")

                # convert semantic mask -> RGB visualization FIRST
                label_map_rgb = Image.fromarray(map_label2RGB(np.array(label_map)))

                titled_images = [
                    add_title(label_map_rgb, "Semantic Mask"),
                    add_title(legend_img, "Legend"),
                    add_title(original_image, "Original Image"),
                ]

                for style, img in zip(styles, images):
                    titled_images.append(add_title(img, f"Generated ({style})"))

                grid = concat_one_line(titled_images,
                                    save_path = f"{vis_dir}/{scene_number}_all_styles.png")

if __name__ == "__main__":
    args = parse_args()  
    main(args)
