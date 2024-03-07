from diffusers import OnnxStableDiffusionPipeline
from typing import Literal

providers = {
    'CPU': 'CPUExecutionProvider',
    'DML': 'DmlExecutionProvider',
    'CUDA': 'n/a'
}

PROVIDER = Literal['CPU','DML','CUDA'] #fuck python



class CustomDiffuser:
    def __init__(self):
        self.pipe = ''
        self.image = None

    def load_model(
            self, 
            path: str = './stable_diffusion_onnx', 
            provider: PROVIDER = 'CPU', 
            safety_checker=None
        ):
        self.pipe = OnnxStableDiffusionPipeline.from_pretrained(path, provider=providers[provider], safety_checker=safety_checker)

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
        #return image
    
    def save_image(self, relative_path: str):
        self.image.save(relative_path)