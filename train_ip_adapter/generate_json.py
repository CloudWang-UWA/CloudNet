import json

output_json = "training_data.json"

ranges = [
    ("cozy", 1, 107),
    ("messy", 108, 209),
    ("classic", 210, 318),
    ("luxurious", 319, 422),
    ("van_gogh", 423, 528),
]

data = []

for style, start, end in ranges:
    for i in range(start, end + 1):
        filename = f"{i:05d}.jpg"
        data.append({
            "image_file": filename,
            "text": f"A bedroom",
            "style": style
        })

with open(output_json, "w") as f:
    json.dump(data, f, indent=4)

print(f"Saved {len(data)} items to {output_json}")