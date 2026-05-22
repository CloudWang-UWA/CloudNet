from safetensors.torch import load_file
import torch
import os

ckpt = "../ip_adapter_models/indoor_raw_output/checkpoint-3000/model.safetensors"
sd = load_file(ckpt)

image_proj_sd = {}
ip_sd = {}

for k in sd:
    if k.startswith("unet"):
        pass
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd[k]

save_dir = "../ip_adapter_models/indoor_models/model_25_02_26"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "ip_adapter.bin")

torch.save(
    {"image_proj": image_proj_sd, "ip_adapter": ip_sd},
    save_path
)