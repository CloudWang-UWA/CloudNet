import os
import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict

generated_dir = "../../image_generation/output"
style_dir = "../../image_generation/style_images"
output_txt = "lpips_results_per_image.txt"
output_avg_txt = "lpips_results_avg.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

styles = ["cozy", "messy", "classic", "luxurious", "van_gogh"]

# LPIPS model
loss_fn = lpips.LPIPS(net='alex').to(device)
loss_fn.eval()

# Transform (normalize to [-1,1])
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5))
])

# Helper: load image
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    return img

# Storage
style_scores = defaultdict(list)
all_scores = []

# Main Loop
with torch.no_grad(), open(output_txt, "w") as f:
    for filename in os.listdir(generated_dir):
        if not filename.endswith(".png"):
            continue

        # Example: classic_art_studio_00038.png
        name = filename.replace(".png", "")
        parts = name.split("_")

        # Robust style detection
        style = None
        for s in styles:
            if name.startswith(s + "_"):  # ensures full style match
                style = s
                break
        if style is None:
            print(f"Style not found in filename {filename}")
            continue

        # Remove style + underscore from start to get remaining, art_studio_00038
        rest = name[len(style) + 1:]

        # [art, studio]
        scene_parts = rest.split("_")[:-1]

        # art_studio
        scene = "_".join(scene_parts) 

        # Construct reference image path
        ref_image_name = f"{style}_{scene}.jpg"
        ref_path = os.path.join(style_dir, scene, ref_image_name)

        gen_path = os.path.join(generated_dir, filename)

        if not os.path.exists(ref_path):
            print(f"Reference not found for {filename}")
            continue

        # Load images
        gen_img = load_image(gen_path)
        ref_img = load_image(ref_path)

        # Compute LPIPS
        score = loss_fn(gen_img, ref_img).item()

        # Save result
        f.write(f"{filename} {score:.6f}\n")

        # Store for averages
        style_scores[style].append(score)
        all_scores.append(score)

# Compute Averages and save to file
with open(output_avg_txt, "w") as f_avg:
    f_avg.write("====== Average LPIPS per Style ======\n")
    for style in styles:
        if len(style_scores[style]) > 0:
            avg = sum(style_scores[style]) / len(style_scores[style])
            f_avg.write(f"{style}: {avg:.6f}\n")
        else:
            f_avg.write(f"{style}: No samples\n")

    if len(all_scores) > 0:
        overall_avg = sum(all_scores) / len(all_scores)
        f_avg.write("\n====== Overall Average ======\n")
        f_avg.write(f"{overall_avg:.6f}\n")