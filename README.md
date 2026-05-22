# CloudNet

CloudNet is a diffusion-based pipeline for style-controllable indoor scene image generation. Given a text prompt, a semantic mask, and a style reference image, the model generates an indoor image that follows the input layout while changing the visual style.

The project combines:

- Stable Diffusion v1.5 as the image generation backbone
- ControlNet for semantic-mask layout conditioning
- a Style IP-Adapter extension for reference-image and learned style-token guidance

## Repository Structure

```text
controlnet/              ControlNet training and dataset utilities
train_ip_adapter/        Style IP-Adapter training code
image_generation_style.py CloudNet generation entry point
ip_adapter_with_style.py Style IP-Adapter inference wrapper
evaluation/              CLIP, aesthetic, LPIPS, and mIoU evaluation scripts
scripts/                 End-to-end workflow and ablation scripts
image_generation/        Prompt folders and style reference images
```

Large assets such as datasets, pretrained weights, trained checkpoints, generated images, W&B logs, and cluster outputs are intentionally not tracked in git.

## Environment

Create a Python environment with PyTorch and install the project requirements:

```bash
pip install -r requirements.txt
```

The original experiments used CUDA, `diffusers==0.24.0`, Stable Diffusion v1.5, and fp16 inference/training.

## Required Assets

Before running generation, prepare the following external assets:

```text
stable-diffusion-v1-5/stable-diffusion-v1-5
stabilityai/sd-vae-ft-mse
cwang_model/controlnet_97_classes
ip_adapter_models/models/image_encoder
ip_adapter_models/indoor_models/model_25_02_26/ip_adapter_with_style.bin
data/ade20k/test/data_ip_test.json
image_generation/original_images
image_generation/semantic_masks
image_generation/style_images
```

These paths are the defaults used by `image_generation_style.sh`. They can be changed in the shell script or passed directly as command-line arguments to `image_generation_style.py`.

The trained project-specific weights and prepared data files are not stored in this repository. Add the download links below when the files are ready:

| Asset | Default path | Link |
| --- | --- | --- |
| 97-class indoor ControlNet | `cwang_model/controlnet_97_classes` | [Google Drive](https://drive.google.com/drive/folders/1iIovFMHSRUaDWqW_M618lsFay0LqDjwX?usp=sharing) |
| Style IP-Adapter checkpoint | `ip_adapter_models/indoor_models/model_25_02_26/ip_adapter_with_style.bin` | [Google Drive](https://drive.google.com/drive/folders/1iIovFMHSRUaDWqW_M618lsFay0LqDjwX?usp=sharing) |
| IP-Adapter image encoder | `ip_adapter_models/models/image_encoder` | [Google Drive](https://drive.google.com/drive/folders/1iIovFMHSRUaDWqW_M618lsFay0LqDjwX?usp=sharing) |
| ADE20K test metadata | `data/ade20k/test/data_ip_test.json` | [Google Drive](https://drive.google.com/drive/folders/1iIovFMHSRUaDWqW_M618lsFay0LqDjwX?usp=sharing) |
| Original test images | `image_generation/original_images` | [Google Drive](https://drive.google.com/drive/folders/1iIovFMHSRUaDWqW_M618lsFay0LqDjwX?usp=sharing) |
| Semantic masks | `image_generation/semantic_masks` | [Google Drive](https://drive.google.com/drive/folders/1iIovFMHSRUaDWqW_M618lsFay0LqDjwX?usp=sharing) |
| Full generated images and evaluation artifacts | external artifact folder | [Google Drive](https://drive.google.com/drive/folders/1iIovFMHSRUaDWqW_M618lsFay0LqDjwX?usp=sharing) |

For public pretrained components, download Stable Diffusion v1.5 and the SD VAE from their original model providers:

- Stable Diffusion v1.5: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- SD VAE: `stabilityai/sd-vae-ft-mse`

## Generate CloudNet Images

Run the default CloudNet generation script:

```bash
bash image_generation_style.sh
```

By default this writes:

```text
image_generation/output
image_generation/prompts
image_generation/visualization
```

The main parameters are:

```bash
python image_generation_style.py \
    --output_folder image_generation/output \
    --prompt_folder image_generation/prompts \
    --vis_folder image_generation/visualization \
    --original_folder image_generation/original_images \
    --semantic_masks_folder image_generation/semantic_masks \
    --style_images_folder image_generation/style_images \
    --base_model_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --controlnet_model_path cwang_model/controlnet_97_classes \
    --vae_model_path stabilityai/sd-vae-ft-mse \
    --image_encoder_path ip_adapter_models/models/image_encoder \
    --ip_adapter_path ip_adapter_models/indoor_models/model_25_02_26/ip_adapter_with_style.bin \
    --scene_label_file data/ade20k/test/data_ip_test.json \
    --num_inference_steps 100 \
    --num_generated_images 1 \
    --scale 0.8
```

## Train ControlNet

The ControlNet training script uses Stable Diffusion v1.5 and ADE20K indoor semantic masks:

```bash
bash controlnet.sh
```

The current training configuration uses:

```text
learning rate: 1e-5
train batch size: 4
gradient accumulation: 2
epochs: 35
checkpoint interval: 2000 steps
validation interval: 1000 steps
mixed precision: fp16
output directory: controlnet/indoor_output/current_model
```

On a Slurm cluster, use:

```bash
sbatch controlnet.slurm
```

## Train the Style IP-Adapter

The Style IP-Adapter training code freezes the Stable Diffusion backbone, CLIP image encoder, original IP-Adapter projection, and IP-Adapter attention layers. It optimizes only the learned style embedding and style projection layer.

```bash
bash scripts/train_style_ip_adapter.sh
```

On a Slurm cluster:

```bash
sbatch scripts/train_style_ip_adapter.slurm
```

## Evaluation

The `evaluation/` folder contains scripts for CLIP-Text, CLIP-Image, CLIP-Aesthetic, LPIPS, and mIoU evaluation. The higher-level workflow scripts are in `scripts/`, including:

```text
scripts/run_style_full_workflow.sh
scripts/run_style_generation_eval.sh
scripts/run_style_miou.sh
scripts/run_style_checkpoint_ablation.sh
scripts/run_style_token_scale_ablation.sh
```

Generated evaluation datasets and metric outputs are ignored by git.

Lightweight summary tables are provided in `results/cloudnet_evaluation_comparison.md`. Full generated images, comparison grids, and intermediate evaluation artifacts should be downloaded separately from the external artifact link listed above.

## Notes

- Model weights and datasets are not included in this repository.
- The default CloudNet configuration uses the trained 97-class indoor ControlNet and the selected Style IP-Adapter checkpoint.
- Paths in the shell scripts are project-relative unless they point to external pretrained assets.
