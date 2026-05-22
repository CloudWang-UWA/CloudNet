import os, json, re, glob

img_dir = "FreestyleNet/image_generation/output"
json_dir = "FreestyleNet/image_generation/json_files"
out_txt = "FreestyleNet/image_generation/prompts/prompts.txt"

lines = []

for name in sorted(os.listdir(img_dir)):
    if not (name.endswith(".jpg") or name.endswith(".png")):
        continue

    m = re.search(r"\d{5}", name)
    if not m:
        continue
    img_id = m.group()

    json_path = os.path.join(json_dir, os.path.splitext(name)[0] + ".json")
    if not os.path.exists(json_path):
        print("No json for", name)
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = data.get("text_label_mapping", {})
    if not mapping:
        print("Empty mapping:", json_path)
        continue

    items = sorted(mapping.items(), key=lambda kv: (kv[1] == -1, kv[1]))
    tokens = [k for k, _ in items]

    lines.append(img_id + " " + " ".join(tokens))

with open(out_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("Saved prompts:", out_txt)
print("Num lines:", len(lines))
