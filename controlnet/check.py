# Obtained from https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py
# Modification: 
#  - Implement Style Swap technique for the validation phase
#  - Apply rare class sampling for the training phase
#  - Convert 3-channel RGB input into 20-channel one-hot embeddings
#  - Customized datasets with transformations


import argparse
import logging
import math
import os
import random
import json
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.pipelines import StableDiffusionControlNetPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from diffusers.models.controlnet import ControlNetModel

# from dataset_Cityscapes import CityScapesDataset
# from dataset_GTA import GTADataset
from dataset_ADE20K import ADE20KDataset
from tools.training_classes import (
    make_one_hot, 
    get_class_stacks,
    map_label2RGB
    )

if is_wandb_available():
    import wandb


logger = get_logger(__name__)


def get_crop_onehot(label_path, resize_ratio=1.0, random_crop_enabled=True):
    data_root = os.environ.get("DATA_ROOT")
    if data_root and not os.path.isabs(label_path):
        label_path = os.path.join(data_root, label_path)
    # print(label_path)
    label_trainId_path = label_path
    # if "_color" in label_path:
    #     label_trainId_path = label_path.replace("_color", "").replace(".png", "_labelTrainIds.png")
    # else:
    #     label_trainId_path = label_path.replace(".png", "_labelTrainIds.png")
    if "_color" in label_path:
        label_trainId_path = label_path.replace("_color", "")
    # else:
        # label_trainId_path = label_path.replace(".png", "_labelTrainIds.png")

    # rgb label color image
    img = Image.open(label_path).convert("RGB")
    # trainable id label image
    label_trainId = Image.open(label_trainId_path)

    crop_size = 512

    w, h = img.size
    new_w, new_h = int(w * resize_ratio), int(h * resize_ratio)
    
    img = img.resize((new_w, new_h), Image.BILINEAR)
    label_trainId = label_trainId.resize((new_w, new_h), Image.NEAREST)
    w, h = new_w, new_h

    if random_crop_enabled:
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
    else:
        x1 = (w -  crop_size) // 2
        y1 = (h -  crop_size) // 2
    
    crop_coords = (x1, y1, x1 + crop_size, y1 + crop_size)
    img = img.crop(crop_coords)
    label_trainId = label_trainId.crop(crop_coords)
    texts = get_class_stacks(label_trainId)

    return img, label_trainId, texts

project_root = Path(__file__).resolve().parents[1]
validate_file = os.environ.get(
    "VALIDATE_FILE",
    str(project_root / "data" / "ade20k" / "validation" / "dataset_file.json"),
)

with open(validate_file, 'r') as f:

    # Initialize empty lists for different keys
    validation_images = []
    
    # Read each line of the file and append values to respective lists
    for line in f:
        # print("11111")
        data = json.loads(line)
        # print(data)                        
        validation_images.append(data.get('conditioning_image'))
        break
    

    image_logs = []
    for i in range(len(validation_images)):
        validation_img = validation_images[i]
        resize_ratio = [0.5, 0.75, 1.0, 1.25, 1.5]
        resize_ratio = resize_ratio[i%len(resize_ratio)] if resize_ratio is not None else 1.0

        # print(validation_img)
        val_img, val_labelTrain, val_prompt = get_crop_onehot(validation_img, resize_ratio, random_crop_enabled=False)

        val_labelTrain_img = Image.fromarray(map_label2RGB(val_labelTrain).astype(np.uint8))

        val_labelTrain_onehot = torch.Tensor(make_one_hot(val_labelTrain))
        val_labelTrain_onehot = torch.unsqueeze(val_labelTrain_onehot.permute(2, 0, 1), 0)

        images = []

        # print(val_prompt)
        # print(val_labelTrain_onehot)


        val_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        unet = UNet2DConditionModel.from_pretrained(
            val_model, subfolder="unet", revision=None
        )
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=97)
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            val_model,
            controlnet=controlnet,
            safety_checker=None,
            revision=None,
            torch_dtype=torch.float32,
        )

        # print("...isinstance", isinstance(controlnet, ControlNetModel))

        for _ in range(2):
            with torch.autocast("cuda"):
                image = pipeline(
                    val_prompt, val_labelTrain_onehot, num_inference_steps=20, generator=None
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": val_labelTrain_img, "images": images, "validation_prompt": val_prompt}
        )

        print ("here")
