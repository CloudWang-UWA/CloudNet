#!/usr/bin/env python3
"""Build the static research page from the checked-in results and original figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
IMAGES = DOCS / "qualitative_results"
SUMMARY = ROOT / "results" / "cloudnet_evaluation_comparison.md"
REPO = "https://github.com/CloudWang-UWA/CloudNet"
STYLES = [("cozy", "Cozy"), ("messy", "Messy"), ("classic", "Classic"),
          ("luxurious", "Luxurious"), ("van_gogh", "Van Gogh")]
SCENES = [("bedroom", "Bedroom"), ("bathroom", "Bathroom"),
          ("living_room", "Living room"), ("kitchen", "Kitchen"),
          ("dining_room", "Dining room")]
METRICS = [("clip_text", "CLIP-Text", 4), ("clip_image", "CLIP-Image", 4),
           ("clip_aes", "CLIP-AES", 4), ("miou", "mIoU", 2)]


def read_sections() -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    current = None
    for line in SUMMARY.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:]
            sections[current] = []
        elif current and line.startswith("|"):
            cells = [x.strip() for x in line.split("|")[1:-1]]
            if not all(set(x) <= {"-", ":", " "} for x in cells):
                sections[current].append(cells)
    return sections


def number(value: str) -> float | None:
    return None if value == "/" else float(value)


def measurements(cells: list[str]) -> dict[str, float | None]:
    return {key: number(cells[i]) for i, (key, _, _) in enumerate(METRICS)}


def picture(relative: str, label: str, *, mask: bool = False, eager: bool = False) -> str:
    original = IMAGES / relative
    if not original.is_file():
        raise FileNotFoundError(original)
    web_relative = Path("assets") / "images" / Path(relative).with_suffix(".webp")
    destination = DOCS / web_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(original) as image:
        image = image.convert("RGB")
        image.thumbnail((720, 540), Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
        image.save(destination, "WEBP", lossless=mask, quality=86, method=6)
        width, height = image.size
    src = web_relative.as_posix()
    full = "qualitative_results/" + relative
    return (f'<button class="image-button" type="button" data-full="{html.escape(full)}" '
            f'data-caption="{html.escape(label)}" aria-label="Enlarge: {html.escape(label)}">'
            f'<img src="{src}" alt="{html.escape(label)}" width="{width}" height="{height}" '
            f'loading="{"eager" if eager else "lazy"}" decoding="async"></button>')


def architecture() -> str:
    relative = "architecture/CloudNet.png"
    label = "CloudNet architecture: semantic ControlNet, Style IP-Adapter, and the diffusion U-Net"
    with Image.open(DOCS / relative) as image:
        width, height = image.size
    return (f'<button class="image-button architecture-image" type="button" data-full="{relative}" '
            f'data-caption="{html.escape(label)}" aria-label="Enlarge: {html.escape(label)}">'
            f'<img src="{relative}" alt="{html.escape(label)}" width="{width}" height="{height}" '
            'loading="eager" decoding="async"></button>')


def teaser() -> str:
    cells = '<div class="column-label">Semantic mask</div>'
    cells += "".join(f'<div class="column-label">{name}</div>' for _, name in STYLES)
    cells += picture("cloudNet/living_room/mask.png", "Living room · input semantic mask", mask=True, eager=True)
    for style, name in STYLES:
        cells += picture(f"cloudNet/living_room/{style}.png", f"Living room · {name} · CloudNet", eager=True)
    return '<div class="teaser-grid">' + cells + "</div>"


def gallery() -> str:
    cells = '<div class="grid-corner">Scene</div><div class="column-label">Semantic mask</div>'
    cells += "".join(f'<div class="column-label">{name}</div>' for _, name in STYLES)
    for index, (scene, label) in enumerate(SCENES):
        cells += f'<div class="row-label"><span>{index + 1:02}</span>{label}</div>'
        cells += picture(f"cloudNet/{scene}/mask.png", f"{label} · input semantic mask", mask=True, eager=True)
        for style, name in STYLES:
            cells += picture(f"cloudNet/{scene}/{style}.png", f"{label} · {name} · CloudNet", eager=True)
    return '<div class="gallery-grid">' + cells + "</div>"


def comparison() -> str:
    result = '<div class="context-row">'
    for filename, name in [("mask", "Input semantic mask"), ("legend", "Semantic classes"), ("original", "Real image")]:
        result += '<figure>' + picture(f"comparison/cloudnet/{filename}.png", name, mask=filename != "original")
        result += f"<figcaption>{name}</figcaption></figure>"
    result += '</div><div class="scroll-area"><div class="comparison-grid">'
    result += '<div></div>' + "".join(f'<div class="column-label">{name}</div>' for _, name in STYLES)
    for model, name in [("cloudnet", "CloudNet"), ("freestylenet", "FreestyleNet")]:
        result += f'<div class="row-label model-label">{name}</div>'
        for style, label in STYLES:
            result += picture(f"comparison/{model}/{style}.png", f"Living room · {label} · {name}")
    return result + "</div></div>"


def metric_table(rows: list[dict], label: str, *, selected: str | None = None) -> str:
    result = f'<table><caption class="sr-only">{html.escape(label)}</caption><thead><tr><th scope="col">{label}</th>'
    result += "".join(f'<th scope="col">{name} <span aria-label="higher is better">↑</span></th>' for _, name, _ in METRICS)
    result += "</tr></thead><tbody>"
    for row in rows:
        chosen = row["name"] == selected
        result += '<tr' + (' class="selected-row"' if chosen else "") + '>'
        result += f'<th scope="row">{html.escape(row["name"])}'
        if chosen:
            result += '<span class="row-badge">Selected</span>'
        result += "</th>"
        for key, _, precision in METRICS:
            value = row[key]
            best = value is not None and value == max((r[key] for r in rows if r[key] is not None), default=None)
            rendered = f"{value:.{precision}f}" if value is not None else '<span aria-label="Not reported">—</span>'
            result += f'<td>{"<strong>" if best else ""}{rendered}{"</strong>" if best else ""}</td>'
        result += "</tr>"
    return result + "</tbody></table>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate generated data and all local page assets.")
    args = parser.parse_args()
    sections = read_sections()
    main_rows = sections["Main Comparison"][1:]
    main = {row[0]: measurements(row[1:5]) for row in main_rows}
    benchmark = [{"name": name, **main[source]} for name, source in [
        ("CloudNet", "CloudNet ckpt6500"), ("FreestyleNet", "FreestyleNet")]]
    ablation = [{"name": name, **main[source]} for name, source in [
        ("Original IP-Adapter", "Original-IP, controlled"), ("Style IP-Adapter (CloudNet)", "CloudNet ckpt6500")]]
    checkpoints = [{"name": row[0].removeprefix("ckpt"), **measurements(row[1:5])}
                   for row in sections["Checkpoint Selection"][1:]]
    scales = [{"name": row[0], **measurements(row[1:5])}
              for row in sections["Style Scale Sensitivity"][1:]]
    style_comparison = []
    for row in sections["Vs FreestyleNet"][2:]:
        style_comparison.append({"style": row[0].replace("Van_gogh", "Van Gogh"),
                                 "cloudnet": measurements([row[1], row[4], row[7], row[10]]),
                                 "freestylenet": measurements([row[2], row[5], row[8], row[11]])})
    payload = {"benchmark": benchmark, "ablation": ablation, "checkpoints": checkpoints,
               "scales": scales, "styleComparison": style_comparison,
               "source": REPO + "/blob/main/results/cloudnet_evaluation_comparison.md",
               "summary_sha256": hashlib.sha256(SUMMARY.read_text(encoding="utf-8").encode("utf-8")).hexdigest()}
    assert len(checkpoints) == len(scales) == 5
    assert len(style_comparison) == 6
    assert all(r[key] is not None for r in checkpoints + scales for key, _, _ in METRICS)
    assert next(r for r in checkpoints if r["name"] == "6500")["clip_aes"] == 6.1487
    if args.check:
        generated = json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))
        assert generated == payload, "Generated data is stale. Run the build script."
        from html.parser import HTMLParser

        class AssetParser(HTMLParser):
            def handle_starttag(self, tag, attrs):
                for key, value in attrs:
                    if key not in {"src", "href", "data-full"} or not value:
                        continue
                    if value.startswith(("http:", "https:", "#", "mailto:")):
                        continue
                    target = DOCS / value.split("#")[0].split("?")[0]
                    assert target.is_file(), f"Missing local resource: {value}"

        page = (DOCS / "index.html").read_text(encoding="utf-8")
        assert "{{" not in page
        AssetParser().feed(page)
        print("Verified: all four experiment tables, 6 style splits, and all page resource paths.")
        return
    ASSETS.mkdir(parents=True, exist_ok=True)
    replacements = {"TEASER": teaser(), "ARCHITECTURE": architecture(), "GALLERY": gallery(), "COMPARISON": comparison(),
                    "BENCHMARK_TABLE": metric_table(benchmark, "Method"),
                    "ABLATION_TABLE": metric_table(ablation, "Adapter"),
                    "CHECKPOINT_TABLE": metric_table(checkpoints, "Training step", selected="6500"),
                    "SCALE_TABLE": metric_table(scales, "Style token scale λ", selected="1.0"),
                    "RESULTS_DATA": json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")}
    template = (ROOT / "scripts" / "project_page_template.html").read_text(encoding="utf-8")
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    (DOCS / "index.html").write_text(template, encoding="utf-8", newline="\n")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    (ASSETS / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    (ASSETS / "evaluation.md").write_text(SUMMARY.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["experiment", "setting", *[name for _, name, _ in METRICS]])
    for section, rows in [("benchmark", benchmark), ("adapter_ablation", ablation), ("checkpoint", checkpoints), ("style_token_scale", scales)]:
        for row in rows:
            writer.writerow([section, row["name"], *[row[key] if row[key] is not None else "" for key, _, _ in METRICS]])
    (ASSETS / "results.csv").write_text(output.getvalue(), encoding="utf-8", newline="\n")
    print(f"Built {DOCS / 'index.html'} from {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
