from diffusers import (
    OnnxStableDiffusionPipeline, 
    OnnxStableDiffusionInpaintPipeline, 
    StableDiffusionPipeline, 
    StableDiffusionInpaintPipeline, 
    ControlNetModel, 
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torch import device
from torch import float16, manual_seed
from typing import Literal
from PIL import Image
import cv2
import random

class CustomDiffuser:
    def __init__(self, provider:Literal['CPUExecutionProvider', 'DmlExecutionProvider', "CUDAExecutionProvider"]='CUDAExecutionProvider'):
        """
        Parameters
        -----------
            provider:str
                str of ['CPUExecutionProvider', 'DmlExecutionProvider', 'CUDAExecutionProvider']

        """
        self.pipe_text2image = None
        self.pipe_inpaint = None
        self.pipe_inpaint_controlnet = None
        self.image = None
        self.sam = None
        self.provider = provider

    def load_model_for_text2image(
            self, 
            path: str = '../stable_diffusion_onnx', 
            safety_checker=None
    ):

        self.pipe_text2image = StableDiffusionPipeline.from_pretrained(path, provider=self.provider,safety_checker=safety_checker)

    def load_model_for_inpainting(
            self, 
            path: str = '../stable_diffusion_onnx_inpainting', 
            safety_checker=None
    ):
        self.pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(path, provider=self.provider, safety_checker=safety_checker)   
        print("TYPE OF PIPELINE:   ", type(self.pipe_inpaint))  

    def load_controlnet_for_inpainting(
            self,
            diffusor_path: str,
            controlnet_path: str,
            safety_checker=None
    ):
        pipe = self.pipe_inpaint_controlnet
        controlnet = ControlNetModel.from_pretrained(controlnet_path, safety_checker=safety_checker)#, torch_dtype=float16)
        #pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        self.pipe_inpaint_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            diffusor_path,
            controlnet=controlnet,
            #torch_dtype=float16 #apparently this doesn't work even thought it's literally in the docu...
            safety_checker=safety_checker
        )
        #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe_inpaint_controlnet.scheduler = UniPCMultistepScheduler.from_config(self.pipe_inpaint_controlnet.scheduler.config)
        #pipe.enable_xformers_memory_efficient_attention() # I actually shouldn't need this on 12GB vram...
        #pipe.to(device("cuda:0"))
        self.pipe_inpaint_controlnet.to(device('cuda:0'))

    def inpaint_with_controlnet(
            self, 
            image: Image.Image, 
            mask: Image.Image,
            control_image: Image.Image,
            height: int, 
            width: int,             
            prompt: str = '', 
            negative: str = '',
            steps: int = 15, 
            cfg: float =  7.5,
            noise: float = 1.0#0.75           
    ):
        image = image.resize((width, height))
        mask = mask.resize((width, height))
        control_image = control_image.resize((width, height))

        print("image sizeeeeeee,    ", image.size)
        print("mask sizeeeeeee,    ", mask.size)
        print("control image sizeeeeeee,    ", control_image.size)
        print("+++++++++++++++++++++++++++++++++++++")

        #random.seed(30)
        #generator = manual_seed(random.random())
        generator = manual_seed(random.randint(0, 99999))
        generated_image = self.pipe_inpaint_controlnet(
            prompt,
            num_inference_steps=20, #pass argument
            generator=generator,
            image=image,
            control_image=control_image,
            mask_image=mask           
        ).images[0]

        return generated_image



    def inpaint_pipe_to_cuda(self):
        self.pipe_inpaint = self.pipe_inpaint.to(device("cuda:0"))
        #self.pipe_text2image = self.pipe_text2image.to(device("cuda:0"))

    def generate_text2image(
        self,
        prompt: str = '', 
        negative: str = '',
        height: int = 512, 
        width: int = 512, 
        steps: int = 10, 
        cfg: float =  7.5
    ):
        self.image = self.pipe(
            prompt, 
            height, 
            width, 
            steps, 
            cfg, 
            negative
        ) \
            .images[0]
        
        return self.image
    
    def inpaint_with_prompt(
            self, 
            image: cv2.typing.MatLike | Image.Image, # sd doesn't really take MatLike so there's no real point having the posibility
            mask: cv2.typing.MatLike | Image.Image,
            height: int, 
            width: int,             
            prompt: str = '', 
            negative: str = '',
            steps: int = 10, 
            cfg: float =  7.5,
            noise: float = 0.75
    ):

        pipe = self.pipe_inpaint

        image = image.resize((width, height))
        mask = mask.resize((width, height))

        print("pipe image shape", image.size)
        print("pipe mask shape", mask.size)

        output = pipe(
            prompt,
            image,
            mask,
            height=height, # you have to pass the dimensions, it does NOT use the same shape for the latent image by default and the output will come out 512x512 otherwise
            width=width,
            #strength=noise,
            guidance_scale=cfg
        )

        return output


    def save_image(self, relative_path: str):
        self.image.save(relative_path)


    # def segment_whole_picture(self, device:PROVIDER='cpu'):
    #     model_name = 'facebook/sam-vit-base'
    #     self.sam = SamModel.from_pretrained(model_name).to(device)
    #     processor = SamProcessor.from_pretrained(model_name)

    #     image_path = './images/spiderman.jpg'
    #     raw_image = Image.open(image_path).convert('RGB')
    #     input_points = [[[256, 256]]]

    #     inputs = processor(
    #         raw_image,
    #         input_points=input_points,
    #         return_tensors='pt'
    #     )\
    #         .to(device)
        
    #     with no_grad():
    #         outputs = self.sam(**inputs)

    #     masks = processor.image_processor.post_process_masks(
    #         outputs.pred_masks.cpu(),
    #         inputs['original_sizes'].cpu(),
    #         inputs['reshaped_input_sizes'].cpu()
    #     )

    #     scores = outputs.iou_scores

    #     mask = masks[0].cpu().numpy()
    #     cv2.imwrite(mask, './images/segmentation_mask.png')

    #     # mask = masks[0][0]
    #     # transform = tv_tr.ToPILImage()
    #     # img = transform(mask)
    #     # img.show()

        
