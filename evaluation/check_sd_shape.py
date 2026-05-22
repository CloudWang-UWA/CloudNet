from diffusers import StableDiffusionPipeline
import torch

model_ckpt = "stable-diffusion-v1-5/stable-diffusion-v1-5"
sd_pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")

prompts = [
    "a photo of an astronaut riding a horse on mars",
    "A high tech solarpunk utopia in the Amazon rainforest",
    "A pikachu fine dining with a view to the Eiffel Tower",
    "A mecha robot in a favela in expressionist style",
    "an insect robot preparing a delicious meal",
    "A small cabin on top of a snowy mountain in the style of Disney, artstation",
]

images = sd_pipeline(prompts, num_images_per_prompt=1, output_type="numpy").images

print(type(images))
print(images.dtype)
print(images.min(), images.max())
print(images.shape)

# <class 'numpy.ndarray'>
# float32
# 0.0 1.0
# (6, 512, 512, 3)