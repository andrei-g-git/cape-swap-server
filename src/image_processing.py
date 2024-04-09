import numpy as np
from torch import Tensor, no_grad
from torch.nn.functional import interpolate
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image

class ImageProcessor:
    def __init__(self) -> None:
        self.depth_processor = None
        self.depth_model = None

    def startup_depth_mapper(self, model_path_or_url:str='Intel/dpt-hybrid-midas'):
        self.depth_processor = DPTImageProcessor.from_pretrained(model_path_or_url)
        self.depth_model = DPTForDepthEstimation.from_pretrained(model_path_or_url)

    def extract_depth_map(self, image:Image.Image):
        inputs = self.depth_processor(images=image, return_tensors='pt')

        with no_grad():
            outputs = self.depth_model(**inputs)
            depth_outputs = outputs.predicted_depth

        depth_outputs_OG_size = interpolate(
            depth_outputs.unsqueeze(1),
            size=image.size[::-1],
            mode='bicubic',
            align_corners=False
        )
        squeezed = depth_outputs_OG_size.\
            squeeze().\
            cpu().\
            numpy()
        denormalized = (squeezed * 255 / np.max(squeezed)).astype('uint8')
        depth_image = Image.fromarray(denormalized)
        return depth_image

        
processor = ImageProcessor()
processor.startup_depth_mapper()
img = processor.extract_depth_map(Image.open('C:/work/py/cape-swap-server/images/in/10.jpg'))
img.save('z-depth.png', 'PNG')


        


