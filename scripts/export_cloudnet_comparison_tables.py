from __future__ import annotations

import csv
import json
import os
from pathlib import Path


EVAL_DIR = Path("evaluation")
OUTPUT_DIR = Path("outputs/cloudnet_metrics")
WRITE_CSV = os.environ.get("WRITE_CSV", "0") == "1"

STYLE_DIRS = ["cozy", "messy", "classic", "luxurious", "van_gogh"]
STYLE_LABELS = {
    "cozy": "Cozy",
    "messy": "Messy",
    "classic": "Classic",
    "luxurious": "Luxurious",
    "van_gogh": "Van_gogh",
    "avg": "Overall",
}
STYLE_ORDER = ["Cozy", "Messy", "Classic", "Luxurious", "Van_gogh", "Overall"]
DELTA_COLUMNS = {3, 6, 9, 12}


RUNS = {
    "cloudnet_6500": {
        "label": "CloudNet ckpt6500",
        "clip_dir": "2026-05-02_style-ipadapter-token5_ckpt6500_controlnet97_100steps_scale0.8",
        "miou_dir": "2026-05-02_style-ipadapter-token5_ckpt6500_controlnet97_100steps_scale0.8_swin-l",
        "note": "Current main model",
    },
    "cloudnet_500": {
        "label": "CloudNet ckpt500",
        "clip_dir": "2026-05-01_style-ipadapter-token5_controlnet97_100steps_scale0.8",
        "miou_dir": "2026-05-01_style-ipadapter-token5_controlnet97_100steps_scale0.8_swin-l",
        "note": "Original token5 checkpoint",
    },
    "normal_ipadapter": {
        "label": "Original-IP, controlled",
        "clip_dir": "2026-05-03_ipadapter_sd15_controlnet97_100steps_scale0.8",
        "miou_dir": "2026-05-03_ipadapter_sd15_controlnet97_100steps_scale0.8_swin-l",
        "note": "100 steps, scale 0.8; preferred Original-IP baseline",
    },
    "pretrained_historical": {
        "label": "Original-IP, historical",
        "clip_dir": "2026-01-09_ipadapter_sd15_controlnet97_70steps",
        "miou_dir": "2026-01-09_ipadapter_sd15_controlnet97_70steps_swin-l",
        "note": "Jan reference only; 70 steps",
    },
    "freestyle": {
        "label": "FreestyleNet",
        "clip_dir": "freestylenet",
        "miou_dir": "freestylenet_baseline_swin-l",
        "note": "No CLIP-I result available",
    },
}

CHECKPOINT_RUNS = [
    ("ckpt500", "2026-05-01_style-ipadapter-token5_controlnet97_100steps_scale0.8", "2026-05-01_style-ipadapter-token5_controlnet97_100steps_scale0.8_swin-l", "Original token5"),
    ("ckpt1500", "2026-05-02_style-ipadapter-token5_ckpt1500_controlnet97_100steps_scale0.8", None, "Style checkpoint ablation"),
    ("ckpt3000", "2026-05-02_style-ipadapter-token5_ckpt3000_controlnet97_100steps_scale0.8", None, "Style checkpoint ablation"),
    ("ckpt5000", "2026-05-02_style-ipadapter-token5_ckpt5000_controlnet97_100steps_scale0.8", None, "Style checkpoint ablation"),
    ("ckpt6500", "2026-05-02_style-ipadapter-token5_ckpt6500_controlnet97_100steps_scale0.8", "2026-05-02_style-ipadapter-token5_ckpt6500_controlnet97_100steps_scale0.8_swin-l", "Best CLIP-AES; current main"),
]

STYLE_SCALE_RUNS = [
    ("0.5", "2026-05-03_style-ipadapter-token5_ckpt6500_styleScale0.5_controlnet97_100steps_scale0.8"),
    ("0.8", "2026-05-03_style-ipadapter-token5_ckpt6500_styleScale0.8_controlnet97_100steps_scale0.8"),
    ("1.0", "2026-05-02_style-ipadapter-token5_ckpt6500_controlnet97_100steps_scale0.8"),
    ("1.2", "2026-05-03_style-ipadapter-token5_ckpt6500_styleScale1.2_controlnet97_100steps_scale0.8"),
    ("1.5", "2026-05-03_style-ipadapter-token5_ckpt6500_styleScale1.5_controlnet97_100steps_scale0.8"),
]


def pct_delta(cloudnet: float | None, baseline: float | None) -> float | None:
    """Relative delta requested for tables: (CloudNet - baseline) / CloudNet."""
    if cloudnet in (None, 0) or baseline is None:
        return None
    return (cloudnet - baseline) / cloudnet


def metric_template() -> dict[str, dict[str, float | None]]:
    return {"clip_text": {}, "clip_i": {}, "clip_aes": {}, "miou": {}}


def read_clip_avg(path: Path, score_name: str) -> dict[str, float]:
    if not path.exists():
        return {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    values = {"Overall": float(lines[1].split("\t")[0])}
    for line in lines[3:]:
        style, _num_images, score = line.split("\t")
        values[STYLE_LABELS[style]] = float(score)
    return values


def read_clip_aes(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("Style"):
            continue
        style, _num_images, score = line.split("\t")
        label = "Overall" if style == "Overall" else STYLE_LABELS[style]
        values[label] = float(score)
    return values


def latest_json(folder: Path) -> Path | None:
    files = sorted(folder.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def read_miou(miou_dir: str | None) -> dict[str, float]:
    if not miou_dir:
        return {}
    root = EVAL_DIR / "mIoU" / miou_dir
    if not root.exists():
        return {}
    values = {}
    for style_dir in ["avg", *STYLE_DIRS]:
        result_file = latest_json(root / style_dir)
        if result_file is None:
            continue
        with result_file.open(encoding="utf-8") as f:
            values[STYLE_LABELS[style_dir]] = float(json.load(f)["mIoU"])
    return values


def read_run(config: dict[str, str]) -> dict[str, dict[str, float | None]]:
    clip_dir = config["clip_dir"]
    metrics = metric_template()
    metrics["clip_text"] = read_clip_avg(EVAL_DIR / "CLIP_score" / clip_dir / "clip_avg_result.txt", "CLIPScore")
    metrics["clip_i"] = read_clip_avg(EVAL_DIR / "CLIP_I" / clip_dir / "clip_avg_result.txt", "CLIP_I_Score")
    metrics["clip_aes"] = read_clip_aes(EVAL_DIR / "CLIP_aes" / clip_dir / "clip_aesthetic_avg_all.txt")
    metrics["miou"] = read_miou(config.get("miou_dir"))
    return metrics


def metric_rows(cloudnet: dict, baseline: dict, baseline_name: str, cloudnet_name: str = "CloudNet") -> list[list]:
    metrics = [
        ("CLIP-Text", "clip_text"),
        ("CLIP-Image", "clip_i"),
        ("CLIP-AES", "clip_aes"),
        ("mIoU", "miou"),
    ]
    rows = [
        ["Style"] + sum(([name, "", ""] for name, _key in metrics), []),
        [""] + sum(([cloudnet_name, baseline_name, "Delta (%)"] for _name, _key in metrics), []),
    ]

    for style in STYLE_ORDER:
        row = [style]
        for _name, key in metrics:
            cn = cloudnet[key].get(style)
            base = baseline[key].get(style)
            row.extend([cn, base, pct_delta(cn, base)])
        rows.append(row)
    return rows


def format_value(value, is_percent: bool = False) -> str:
    if value is None:
        return "/"
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.2%}" if is_percent else f"{value:.4f}"
    return str(value)


def formatted_rows(rows: list[list], percent_cols: set[int] | None = None) -> list[list[str]]:
    percent_cols = percent_cols or set()
    return [
        [format_value(value, is_percent=(r >= 2 and c in percent_cols)) for c, value in enumerate(row)]
        for r, row in enumerate(rows)
    ]


def write_csv(path: Path, rows: list[list], percent_cols: set[int] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(formatted_rows(rows, percent_cols))


def write_combined_csv(path: Path, tables: list[tuple[str, list[list], set[int]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for title, rows, percent_cols in tables:
            writer.writerow([title])
            writer.writerows(formatted_rows(rows, percent_cols))
            writer.writerow([])


def markdown_table(rows: list[list], percent_cols: set[int] | None = None) -> str:
    rows = formatted_rows(rows, percent_cols)
    if not rows:
        return ""
    widths = [max(len(row[c]) if c < len(row) else 0 for row in rows) for c in range(max(len(r) for r in rows))]

    def render(row: list[str]) -> str:
        return "| " + " | ".join((row[c] if c < len(row) else "").ljust(widths[c]) for c in range(len(widths))) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(rows[0]), separator, *[render(row) for row in rows[1:]]])


def write_markdown(path: Path, tables: list[tuple[str, list[list], set[int]]]) -> None:
    lines = [
        "# CloudNet Evaluation Comparison",
        "",
        "Relative delta uses `(CloudNet - baseline) / CloudNet`.",
        "Missing values are shown as `/`.",
        "",
    ]
    for title, rows, percent_cols in tables:
        lines.extend([f"## {title}", "", markdown_table(rows, percent_cols), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = {name: read_run(config) for name, config in RUNS.items()}
    cloudnet = runs["cloudnet_6500"]

    vs_freestyle = metric_rows(cloudnet, runs["freestyle"], "FreestyleNet")
    vs_normal = metric_rows(cloudnet, runs["normal_ipadapter"], "Original-IP", "Style-IP")
    vs_historical = metric_rows(cloudnet, runs["pretrained_historical"], "Historical Original-IP", "Style-IP")

    main_comparison = [["Method", "CLIP-Text", "CLIP-Image", "CLIP-AES", "mIoU", "Note"]]
    for key in ["normal_ipadapter", "pretrained_historical", "freestyle", "cloudnet_500", "cloudnet_6500"]:
        config = RUNS[key]
        metrics = runs[key]
        main_comparison.append(
            [
                config["label"],
                metrics["clip_text"].get("Overall"),
                metrics["clip_i"].get("Overall"),
                metrics["clip_aes"].get("Overall"),
                metrics["miou"].get("Overall"),
                config["note"],
            ]
        )

    checkpoint_selection = [["Checkpoint", "CLIP-Text", "CLIP-Image", "CLIP-AES", "mIoU", "Comment"]]
    for label, clip_dir, miou_dir, comment in CHECKPOINT_RUNS:
        config = {"clip_dir": clip_dir, "miou_dir": miou_dir}
        metrics = read_run(config)
        checkpoint_selection.append(
            [
                label,
                metrics["clip_text"].get("Overall"),
                metrics["clip_i"].get("Overall"),
                metrics["clip_aes"].get("Overall"),
                metrics["miou"].get("Overall"),
                comment,
            ]
        )

    style_scale_sensitivity = [["Style token scale", "CLIP-Text", "CLIP-Image", "CLIP-AES", "mIoU", "Comment"]]
    for scale, run_name in STYLE_SCALE_RUNS:
        metrics = read_run({"clip_dir": run_name, "miou_dir": f"{run_name}_swin-l"})
        comment = "Default/main ckpt6500 setting" if scale == "1.0" else "Inference-only sensitivity"
        style_scale_sensitivity.append(
            [
                scale,
                metrics["clip_text"].get("Overall"),
                metrics["clip_i"].get("Overall"),
                metrics["clip_aes"].get("Overall"),
                metrics["miou"].get("Overall"),
                comment,
            ]
        )

    miou_stylewise = [["Method", "Overall", "Cozy", "Messy", "Classic", "Luxurious", "Van_gogh"]]
    for key in ["normal_ipadapter", "pretrained_historical", "freestyle", "cloudnet_500", "cloudnet_6500"]:
        metrics = runs[key]["miou"]
        miou_stylewise.append(
            [
                RUNS[key]["label"],
                metrics.get("Overall"),
                metrics.get("Cozy"),
                metrics.get("Messy"),
                metrics.get("Classic"),
                metrics.get("Luxurious"),
                metrics.get("Van_gogh"),
            ]
        )

    notes = [
        ["Item", "Value"],
        ["CloudNet main model", RUNS["cloudnet_6500"]["clip_dir"]],
        ["Delta convention", "(CloudNet - baseline) / CloudNet"],
        ["FreestyleNet CLIP-Image", "Not available in current evaluation outputs"],
        ["Normal IP-Adapter baseline", "Use the controlled 2026-05-03 100-step/scale-0.8 run for formal comparison"],
        ["Historical normal IP-Adapter", "Keep the Jan 70-step run as reference only"],
    ]

    tables = [
        ("Main Comparison", main_comparison, set()),
        ("Vs FreestyleNet", vs_freestyle, DELTA_COLUMNS),
        ("Style-IP vs Original-IP Controlled", vs_normal, DELTA_COLUMNS),
        ("Style-IP vs Original-IP Historical", vs_historical, DELTA_COLUMNS),
        ("Checkpoint Selection", checkpoint_selection, set()),
        ("Style Scale Sensitivity", style_scale_sensitivity, set()),
        ("mIoU Stylewise", miou_stylewise, set()),
        ("Notes", notes, set()),
    ]

    write_markdown(OUTPUT_DIR / "cloudnet_evaluation_comparison.md", tables)
    if WRITE_CSV:
        write_combined_csv(OUTPUT_DIR / "cloudnet_evaluation_comparison.csv", tables)
        write_csv(OUTPUT_DIR / "cloudnet_vs_freestylenet.csv", vs_freestyle, DELTA_COLUMNS)
        write_csv(OUTPUT_DIR / "cloudnet_vs_normal_ipadapter_controlled.csv", vs_normal, DELTA_COLUMNS)
        write_csv(OUTPUT_DIR / "cloudnet_vs_normal_ipadapter_historical.csv", vs_historical, DELTA_COLUMNS)
        write_csv(OUTPUT_DIR / "cloudnet_main_comparison.csv", main_comparison)
        write_csv(OUTPUT_DIR / "cloudnet_checkpoint_selection.csv", checkpoint_selection)
        write_csv(OUTPUT_DIR / "cloudnet_miou_stylewise.csv", miou_stylewise)

    xlsx_path = OUTPUT_DIR / "cloudnet_evaluation_comparison.xlsx"
    if xlsx_path.exists():
        try:
            xlsx_path.unlink()
        except PermissionError:
            print(f"Warning: could not remove open/locked file: {xlsx_path}")

    output_kind = "Markdown and CSV" if WRITE_CSV else "Markdown"
    print(f"Wrote {output_kind} tables to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
