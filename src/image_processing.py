import gc
import os, sys
from typing import Tuple
import numpy as np
from torch import no_grad, cuda
from torch.nn.functional import interpolate
from transformers import DPTImageProcessor, DPTForDepthEstimation, AutoImageProcessor
from PIL import Image
from huggingface_hub import hf_hub_download
import tempfile
from torch.hub import get_dir
from torch import device
#import torch
#sys.path.append(os.getcwd() + '/..')
from comfyui_controlnet_aux.src.controlnet_aux.normalbae import NormalBaeDetector

class ImageProcessor:
    def __init__(self) -> None:
        self.depth_processor = None
        self.depth_model = None
        self.normals_pipe = None
        self.device = device('cuda:0')

    def startup_depth_mapper(self, model_path_or_url:str='Intel/dpt-hybrid-midas'):
        #self.depth_processor = DPTImageProcessor.from_pretrained(model_path_or_url)
        self.depth_processor = AutoImageProcessor.from_pretrained(model_path_or_url)
        self.depth_model = DPTForDepthEstimation.from_pretrained(model_path_or_url)

    def startup_normal_mapper(self, model_path_or_url:str='lllyasviel/Annotators', file_name:str='scannet.pt', device:device=device('cuda:0')):
    #def startup_normal_mapper(self, model_path_or_url:str='C:/work/py/cape-swap-server/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators', file_name:str='scannet.pt', device:device=device('cuda:0')):

        # model_path = hf_hub_download(  # no good, I'd need to copy paste a whole slew of models that I don't understand, I'll just clone comfyui_controlnet_aux
        #     repo_id=model_path_or_url,
        #     cache_dir=tempfile.gettempdir(),
        #     local_dir=os.path.join(get_dir(), 'checkpoints'),
        #     subfolder='',
        #     filename=file_name,
        #     local_dir_use_symlinks=True,
        #     resume_download=True,
        #     etag_timeout=100,
        #     repo_type='model'
        # )
        self.device = device
        self.normals_pipe = NormalBaeDetector.from_pretrained(model_path_or_url, file_name).to('cpu')#device)

    def extract_normal_map(self, image:Image.Image):
        np_image = np.asarray(image, dtype=np.uint8)
        denormalized_image = np_image * 255
        self.normals_pipe.to(self.device)
        output = self.normals_pipe(denormalized_image, output_type='np', detect_resolution=512)
        normal_image = Image.fromarray(output)

        #del self.normals_pipe

        gc.collect()
        with no_grad():
            cuda.empty_cache()

        return normal_image


    def extract_depth_map(self, image:Image.Image):
        with no_grad():
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

        del self.depth_model

        return depth_image

    def resize_image_keep_aspect(self, image:Image.Image, size:Tuple[int, int]):
        resized = image.copy()
        resized.thumbnail(size, Image.Resampling.LANCZOS)

        return resized


        
# processor = ImageProcessor()
# #processor.startup_depth_mapper()
# processor.startup_normal_mapper()
# img = processor.extract_normal_map(Image.open('C:/work/py/cape-swap-server/images/in/10.jpg'))
# img.save('z-normals.png', 'PNG')


        


