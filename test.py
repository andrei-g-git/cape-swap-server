from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained("C:/work/py/models/runwayml/stable-diffusion-1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
image = pipe("photo of a woman taking a selfie, yoga pants, blond hair").images[0]
image.save("zzz-delete-this.png")


# # from custom_diffusers import CustomDiffuser

# # diffuser = CustomDiffuser()

# # diffuser.segment_whole_picture()




# from transformers import SamModel, SamProcessor
# from PIL import Image
# from torch import no_grad
# import cv2
# import torch
# from torchvision.utils import draw_segmentation_masks
# from torchvision.io import read_image





# model_name = 'facebook/sam-vit-base'
# sam = SamModel.from_pretrained(model_name).to('cpu')
# processor = SamProcessor.from_pretrained(model_name)

# image_path = './images/spiderman.jpg'
# raw_image = Image.open(image_path).convert('RGB')
# input_points = [[[256, 256]]]

# inputs = processor(
#     raw_image,
#     input_points=input_points,
#     return_tensors='pt'
# )\
#     .to('cpu')

# with no_grad():
#     outputs = sam(**inputs)

# masks = processor.image_processor.post_process_masks(
#     outputs.pred_masks.cpu(),
#     inputs['original_sizes'].cpu(),
#     inputs['reshaped_input_sizes'].cpu()
# )

# scores = outputs.iou_scores

# print('masks thing... \n', masks[0])
# print('shape:  \n', masks[0].shape)
# spiderman = read_image('images/spiderman.jpg')
# msk = torch.reshape(masks[0], (3, 512, 512))
# draw_segmentation_masks(spiderman, msk)

# # mask = masks[0].cpu().numpy()
# # cv2.imwrite(mask, './images/segmentation_mask.png')
