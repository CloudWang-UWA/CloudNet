"""
Code to download clip_aes score model
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

AES_DIR = os.environ.get(
    "AES_DIR",
    str(Path(__file__).resolve().parent / "models"),
)

def get_aesthetic_model(clip_model="vit_l_14"):
    os.makedirs(AES_DIR, exist_ok=True)

    path_to_model = os.path.join(
        AES_DIR, f"sa_0_4_{clip_model}_linear.pth"
    )
    if not os.path.exists(path_to_model):
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

if __name__ == "__main__":
    model = get_aesthetic_model("vit_l_14")
    print("Aesthetic model ready")
