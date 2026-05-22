#!/usr/bin/env python3
import argparse
from pathlib import Path


STYLES = ["cozy", "messy", "classic", "luxurious", "van_gogh"]


def parse_clip_avg(path: Path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    overall = float(lines[1].split("\t")[0])
    values = {"Overall": overall}
    for line in lines[3:]:
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2] != "NA":
            values[parts[0]] = float(parts[2])
    return values


def parse_aes_avg(path: Path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("Style"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            values[parts[0]] = float(parts[2])
    return values


def row(metric, label, new_values, old_values):
    keys = ["Overall"] + STYLES
    print(f"\n## {metric}: {label}")
    print("split\told\tnew\tdelta\tdelta_%")
    for key in keys:
        if key not in new_values or key not in old_values:
            continue
        old = old_values[key]
        new = new_values[key]
        delta = new - old
        pct = (delta / old * 100.0) if old else 0.0
        print(f"{key}\t{old:.6f}\t{new:.6f}\t{delta:+.6f}\t{pct:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Compare CLIP metrics between two experiment runs.")
    parser.add_argument("--new-run", required=True, help="New run directory name under evaluation metric folders.")
    parser.add_argument(
        "--old-run",
        default="2026-02-25_style-ipadapter_controlnet97_100steps_scale0.8",
        help="Baseline run directory name under evaluation metric folders.",
    )
    parser.add_argument("--evaluation-root", default="evaluation")
    args = parser.parse_args()

    root = Path(args.evaluation_root)
    old = args.old_run
    new = args.new_run

    paths = {
        "CLIP-Text": (
            root / "CLIP_score" / old / "clip_avg_result.txt",
            root / "CLIP_score" / new / "clip_avg_result.txt",
            parse_clip_avg,
        ),
        "CLIP-I": (
            root / "CLIP_I" / old / "clip_avg_result.txt",
            root / "CLIP_I" / new / "clip_avg_result.txt",
            parse_clip_avg,
        ),
        "CLIP-AES": (
            root / "CLIP_aes" / old / "clip_aesthetic_avg_all.txt",
            root / "CLIP_aes" / new / "clip_aesthetic_avg_all.txt",
            parse_aes_avg,
        ),
    }

    for metric, (old_path, new_path, parser_fn) in paths.items():
        missing = [str(p) for p in [old_path, new_path] if not p.exists()]
        if missing:
            print(f"\n## {metric}: missing files")
            for path in missing:
                print(path)
            continue
        row(metric, f"{old} -> {new}", parser_fn(new_path), parser_fn(old_path))


if __name__ == "__main__":
    main()
