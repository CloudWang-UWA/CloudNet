# CloudNet project page

This directory is a static, buildless GitHub Pages site. The intended public URL is:

https://cloudwang-uwa.github.io/CloudNet/

## Content and sources

- `index.html`: generated page; edit `../scripts/project_page_template.html` instead.
- `assets/site.css` and `assets/site.js`: presentation and browser interactions.
- `architecture/CloudNet.png`: original architecture diagram, displayed in the Method section before qualitative results and available in the image viewer.
- `assets/images/`: generated WebP display copies, linked to full-resolution PNGs in `qualitative_results/`.
- `assets/results.json`, `assets/results.csv`, and `assets/evaluation.md`: generated experiment data and summary downloads.
- `../results/cloudnet_evaluation_comparison.md`: authoritative result summary.

The page opens with the formal project title and author/supervisor, followed by the Abstract with a living room example strip, then the method and architecture diagram. The example strip reuses the living room images without removing their row from the later complete gallery.

The complete qualitative grid reproduces the five main scenes in dissertation Figure 4.1. The model-comparison grid uses the living room comparison from Figure 4.2. Other scene and backup assets remain in `qualitative_results/` but are not used in these page figures.

## Rebuild and validate

With Python and Pillow installed, run from the repository root:

```bash
python scripts/build_project_page.py
python scripts/build_project_page.py --check
node --check docs/assets/site.js
python -m http.server 8765 --directory docs
```

Open http://localhost:8765/ to preview. No Node packages, external fonts, or third-party JavaScript are required by the page. Images can be enlarged with a keyboard or pointer; Escape closes the image viewer. Wide image grids and data tables can be scrolled on smaller screens.

The build reads all displayed metrics from the result summary. Rebuild after changing that summary; do not edit the generated JSON, CSV, download copy, or HTML tables independently.

## Enable GitHub Pages

A repository administrator needs to configure **Settings → Pages → Build and deployment**:

- Source: **Deploy from a branch**
- Branch: **main**
- Folder: **/docs**

Save the configuration. Once GitHub completes its Pages deployment, subsequent commits to `main` update the website automatically. `.nojekyll` keeps the static page and assets from being processed by Jekyll.

## Experimental interpretation

The controlled adapter comparison is the main component ablation. Style-token-scale results are an inference sensitivity study with checkpoint 6500 held fixed. The checkpoint table documents model selection: 6500 has the best aesthetic score among the tested checkpoints, while 500 has the best mIoU and CLIP-Text. These studies should not be combined into a claim that one setting wins on every metric.

The main style-token scale is 1.0, independently of the IP-Adapter scale of 0.8. The checkpoint labels refer to training steps, not denoising steps. FreestyleNet's CLIP-Image score is not reported in this evaluation.
