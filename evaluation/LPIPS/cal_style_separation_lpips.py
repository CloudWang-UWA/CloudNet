import argparse
import csv
import itertools
import os
from collections import defaultdict

import lpips
import torch
import torchvision.transforms as transforms
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute pairwise LPIPS distances between style datasets."
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Dataset root containing one subdirectory per style.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for LPIPS style-separation outputs.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["cozy", "messy", "classic", "luxurious", "van_gogh"],
    )
    parser.add_argument("--image_suffix", default=".png")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    return parser.parse_args()


def content_key(filename, style):
    stem, _ = os.path.splitext(filename)
    prefix = style + "_"
    if not stem.startswith(prefix):
        return None
    return stem[len(prefix):]


def load_image(path, transform, device):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net=args.net).to(device)
    loss_fn.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    by_key = defaultdict(dict)
    for style in args.styles:
        style_dir = os.path.join(args.base_dir, style)
        if not os.path.isdir(style_dir):
            raise FileNotFoundError(f"Style directory not found: {style_dir}")

        for filename in sorted(os.listdir(style_dir)):
            if not filename.lower().endswith(args.image_suffix.lower()):
                continue
            key = content_key(filename, style)
            if key is None:
                continue
            by_key[key][style] = os.path.join(style_dir, filename)

    pair_scores = defaultdict(list)
    style_scores = defaultdict(list)
    all_scores = []
    per_pair_path = os.path.join(args.output_dir, "lpips_style_separation_per_pair.csv")

    with torch.no_grad(), open(per_pair_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["content_key", "style_a", "style_b", "lpips"])

        for key in sorted(by_key):
            available_styles = [style for style in args.styles if style in by_key[key]]
            for style_a, style_b in itertools.combinations(available_styles, 2):
                image_a = load_image(by_key[key][style_a], transform, device)
                image_b = load_image(by_key[key][style_b], transform, device)
                score = loss_fn(image_a, image_b).item()

                pair_name = f"{style_a}__{style_b}"
                pair_scores[pair_name].append(score)
                style_scores[style_a].append(score)
                style_scores[style_b].append(score)
                all_scores.append(score)
                writer.writerow([key, style_a, style_b, f"{score:.6f}"])

    summary_csv = os.path.join(args.output_dir, "lpips_style_separation_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "count", "mean_lpips"])
        for pair_name in sorted(pair_scores):
            scores = pair_scores[pair_name]
            writer.writerow([pair_name, len(scores), f"{sum(scores) / len(scores):.6f}"])
        for style in args.styles:
            scores = style_scores[style]
            if scores:
                writer.writerow([f"{style}_vs_others", len(scores), f"{sum(scores) / len(scores):.6f}"])
        if all_scores:
            writer.writerow(["overall", len(all_scores), f"{sum(all_scores) / len(all_scores):.6f}"])

    summary_txt = os.path.join(args.output_dir, "lpips_style_separation_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("====== Pairwise Style Separation LPIPS ======\n")
        for pair_name in sorted(pair_scores):
            scores = pair_scores[pair_name]
            f.write(f"{pair_name}: {sum(scores) / len(scores):.6f} ({len(scores)} pairs)\n")
        f.write("\n====== Style vs Others ======\n")
        for style in args.styles:
            scores = style_scores[style]
            if scores:
                f.write(f"{style}: {sum(scores) / len(scores):.6f} ({len(scores)} pairs)\n")
        if all_scores:
            f.write("\n====== Overall ======\n")
            f.write(f"{sum(all_scores) / len(all_scores):.6f} ({len(all_scores)} pairs)\n")

    print(f"Saved per-pair LPIPS: {per_pair_path}")
    print(f"Saved summary: {summary_txt}")


if __name__ == "__main__":
    main()
