import json
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm
from training_classes import get_class_stacks


def get_dataset_file(folder):
    imgs_path = folder + "/images"
    file_output = folder + "/dataset_file.json"
    data = []
    for img_path in tqdm(list(sorted(glob(f"{imgs_path}/*.jpg")))):
        
        label_path = img_path.replace("images", "labels").replace(".jpg", ".png")
        # label_train_path = label_path.replace(".png", "_labelTrainIds.png")

        sentence = get_class_stacks(label_path)
        data.append({"text":sentence, "image": img_path, "conditioning_image":label_path})
    
    with open(file_output, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')

        
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    folder = os.environ.get("ADE20K_TEST_DIR", str(project_root / "data" / "ade20k" / "test"))
    get_dataset_file(folder)
