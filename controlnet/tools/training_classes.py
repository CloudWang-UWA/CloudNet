# Author: Cloud Wang. Last Modified: 24/12/2025

import json
import os.path as osp
from PIL import Image
import numpy as np
import torch

# Map ADE20K label IDs to your selected indoor classes
ADE20K_ID_TO_CLASS = {
    1: "wall",
    4: "floor",
    6: "ceiling",
    8: "bed",
    9: "windowpane",
    11: "cabinet",
    13: "person",
    15: "door",
    16: "table",
    18: "plant",
    19: "curtain",
    20: "chair",
    23: "painting",
    24: "sofa",
    25: "shelf",
    28: "mirror",
    29: "rug",
    31: "armchair",
    32: "seat",
    34: "desk",
    36: "wardrobe",
    37: "lamp",
    38: "bathtub",
    39: "railing",
    40: "cushion",
    41: "base",
    42: "box",
    43: "column",
    45: "chest of drawers",
    46: "counter",
    48: "sink",
    50: "fireplace",
    51: "refrigerator",
    54: "stairs",
    56: "case",
    57: "pool table",
    58: "pillow",
    59: "screen door",
    60: "stairway",
    63: "bookcase",
    64: "blind",
    65: "coffee table",
    66: "toilet",
    67: "flower",
    68: "book",
    70: "bench",
    71: "countertop",
    72: "stove",
    74: "kitchen island",
    75: "computer",
    76: "swivel chair",
    78: "bar",
    79: "arcade machine",
    82: "towel",
    83: "light",
    86: "chandelier",
    90: "television receiver",
    93: "apparel",
    94: "pole",
    96: "bannister",
    97: "escalator",
    98: "ottoman",
    99: "bottle",
    100: "buffet",
    101: "poster",
    108: "washer",
    109: "plaything",
    111: "stool",
    112: "barrel",
    113: "basket",
    116: "bag",
    118: "cradle",
    119: "oven",
    120: "ball",
    121: "food",
    122: "step",
    125: "microwave",
    126: "pot",
    130: "dishwasher",
    131: "screen",
    132: "blanket",
    133: "sculpture",
    134: "hood",
    135: "sconce",
    136: "vase",
    138: "tray",
    139: "ashcan",
    140: "fan",
    142: "crt screen",
    143: "plate",
    144: "monitor",
    145: "bulletin board",
    146: "shower",
    147: "radiator",
    148: "glass",
    149: "clock"
}

ADE20K_CLASS_TO_INDEX = {k: i for i, k in enumerate(ADE20K_ID_TO_CLASS.keys())}

def get_cs_classes():
    """ADE20K 96 indoor classes."""
    return [
        "wall",
        "floor",
        "ceiling",
        "bed",
        "windowpane",
        "cabinet",
        "person",
        "door",
        "table",
        "plant",
        "curtain",
        "chair",
        "painting",
        "sofa",
        "shelf",
        "mirror",
        "rug",
        "armchair",
        "seat",
        "desk",
        "wardrobe",
        "lamp",
        "bathtub",
        "railing",
        "cushion",
        "base",
        "box",
        "column",
        "chest of drawers",
        "counter",
        "sink",
        "fireplace",
        "refrigerator",
        "stairs",
        "case",
        "pool table",
        "pillow",
        "screen door",
        "stairway",
        "bookcase",
        "blind",
        "coffee table",
        "toilet",
        "flower",
        "book",
        "bench",
        "countertop",
        "stove",
        "kitchen island",
        "computer",
        "swivel chair",
        "bar",
        "arcade machine",
        "towel",
        "light",
        "chandelier",
        "television receiver",
        "apparel",
        "pole",
        "bannister",
        "escalator",
        "ottoman",
        "bottle",
        "buffet",
        "poster",
        "washer",
        "plaything",
        "stool",
        "barrel",
        "basket",
        "bag",
        "cradle",
        "oven",
        "ball",
        "food",
        "step",
        "microwave",
        "pot",
        "dishwasher",
        "screen",
        "blanket",
        "sculpture",
        "hood",
        "sconce",
        "vase",
        "tray",
        "ashcan",
        "fan",
        "crt screen",
        "plate",
        "monitor",
        "bulletin board",
        "shower",
        "radiator",
        "glass",
        "clock",
    ]

def remap_label_map(label_map):
    """Convert raw ADE20K IDs to contiguous indices (0–95), and others to 96."""
    label_map = np.array(label_map)
    remapped = np.full_like(label_map, fill_value=96)  # default ignore
    for ade_id, idx in ADE20K_CLASS_TO_INDEX.items():
        remapped[label_map == ade_id] = idx
    return remapped

def get_cs_palette():
    return [
        [120, 120, 120], [80, 50, 50], [120, 120, 80], [204, 5, 255], [230, 230, 230], [224, 5, 255], [150, 5, 61], [8, 255, 51], [255, 6, 82], [204, 255, 4], [255, 51, 7], [204, 70, 3], [255, 6, 51], [11, 102, 255], [255, 7, 71], [220, 220, 220], [255, 9, 92], [8, 255, 214], [7, 255, 224], [10, 255, 71], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [6, 51, 255], [235, 12, 255], [0, 163, 255], [250, 10, 15], [20, 255, 0], [255, 224, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], [0, 255, 245], [0, 61, 255], [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 255, 41], [0, 255, 173], [10, 0, 255], [0, 255, 153], [255, 92, 0], [255, 0, 102], [255, 173, 0], [0, 31, 255], [0, 255, 194], [0, 112, 255], [51, 0, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0], [143, 255, 0], [184, 0, 255], [255, 0, 31], [0, 214, 255], [255, 0, 112], [92, 255, 0], [70, 184, 160], [153, 0, 255], [71, 255, 0], [255, 0, 163], [255, 204, 0], [255, 0, 143], [255, 0, 235], [245, 0, 255], [214, 255, 0], [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204], [41, 255, 0], [173, 0, 255], [0, 245, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0]
    ]

def get_class_stacks(label_map):
    if isinstance(label_map, str):
        label_map = np.array(Image.open(label_map))
    elif isinstance(label_map, Image.Image):
        label_map = np.array(label_map)
    else:
        label_map = label_map
    
    label_map = remap_label_map(label_map)

    labels = np.unique(label_map)
    cs_classes = get_cs_classes()
    sentence = [cs_classes[i] for i in labels if i!=96]
    sentence = " ".join(sentence)
    return sentence


def make_one_hot(label_map):
    label_map = np.array(label_map) if not isinstance(
        label_map, np.ndarray) else label_map
    label_map = remap_label_map(label_map)
    num_classes = 97

    one_hot = np.eye(num_classes)[label_map]

    return one_hot


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


def get_label_stats(label_map):
    label_map = np.array(label_map) if not isinstance(
        label_map, np.ndarray) else label_map
    label_map = remap_label_map(label_map)
    labels = np.unique(label_map)
    cs_classes = get_cs_classes()
    label_stats = {}
    
    for i in range(len(cs_classes)):
        label_stats[cs_classes[i]] = np.sum(label_map == i)    
    label_stats["others"] = np.sum(label_map == 96)

    return label_stats


def map_label2RGB(label_map):
    """
    args:
        label_map: (H, W)
    return:
        color_map: (H, W, 3), numpy array    
    """
    
    label_map = np.array(label_map) if not isinstance(label_map, np.ndarray) else label_map
    label_map = remap_label_map(label_map)
    palette = np.array(get_cs_palette())

    color_map = np.zeros((label_map.shape[0], label_map.shape[1], 3))
    for label in range(0, 96):
        color_map[label_map == label] = palette[label]

    color_map = color_map.astype(np.uint8)

    return color_map


def map_RGB2label(color_map):
    """
    args:
        color_map: (H, W, 3), numpy array        
    return:
        label_map: (H, W), numpy array
    """

    palette = np.array(get_cs_palette())
    palette_dict = {tuple(color): label for label, color in enumerate(palette)}

    label_map = np.zeros((color_map.shape[0], color_map.shape[1]), dtype=np.uint8)

    for y in range(color_map.shape[0]):
        for x in range(color_map.shape[1]):
            rgb = tuple(color_map[y, x])
            label_map[y, x] = palette_dict.get(rgb, 96)  # Default to label 96 if color not in palette

    return label_map

def get_text_label_mapping(label_map):
    if isinstance(label_map, str):
        label_map = np.array(Image.open(label_map))
    elif isinstance(label_map, Image.Image):
        label_map = np.array(label_map)

    remapped = remap_label_map(label_map)
    labels = np.unique(remapped)

    cs_classes = get_cs_classes()
    class2ade = {v: k for k, v in ADE20K_ID_TO_CLASS.items()}

    mapping = {}
    for i in labels:
        if i != 96:
            name = cs_classes[int(i)]
            mapping[name] = int(class2ade[name])

    return mapping
