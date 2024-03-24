from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionInpaintPipeline
from typing import Literal
from PIL import Image
import cv2


class CustomDiffuser:
    def __init__(self, provider:Literal['CPUExecutionProvider', 'DmlExecutionProvider']='CPUExecutionProvider'):
        """
        Parameters
        -----------
            provider:str
                str of ['CPUExecutionProvider', 'DmlExecutionProvider']

        """
        self.pipe_text2image = None
        self.pipe_inpaint = None
        self.image = None
        self.sam = None
        self.provider = provider

    def load_model_for_text2image(
            self, 
            path: str = '../stable_diffusion_onnx', 
            safety_checker=None
    ):

        self.pipe_text2image = OnnxStableDiffusionPipeline.from_pretrained(path, provider=self.provider,safety_checker=safety_checker)

    def load_model_for_inpainting(
            self, 
            path: str = '../stable_diffusion_onnx_inpainting', 
            safety_checker=None
    ):
        self.pipe_inpaint = OnnxStableDiffusionInpaintPipeline.from_pretrained(path, provider=self.provider, revision='onnx', safety_checker=safety_checker)        

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

        output_image = pipe(
            prompt,
            image,
            mask,
            #strength=noise,
            guidance_scale=cfg
        )

        return output_image


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

        
