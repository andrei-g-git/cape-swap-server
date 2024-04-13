# import torch
# from diffusers import StableDiffusionPipeline

# model = StableDiffusionPipeline.from_pretrained(
#     'imagepipeline/Realistic-Vision-V6.0',
#     torch_dtype=torch.float16, 
#     safety_checker=False,
# )

#torch.save(model.state_dict())

from PIL import Image

image = Image.open('C:/work/py/cape-swap-server/images/in/1.jpg')

image.thumbnail((768, 768), Image.Resampling.LANCZOS)

image.save('zzz-delete.png', "PNG")