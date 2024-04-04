from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image
from torch import device, load
import json
from custom_diffusers import CustomDiffuser
from masking import Masking
from bisnet import BiSeNet

app = Flask(__name__)
CORS(app)

app.debug = True

class FlaskSDServer:
    def __init__(self,  diffusion_handler: CustomDiffuser, masker: Masking):
        self.sd = diffusion_handler
        self.masker = masker

    def initialize_diffusor_and_masking(self):
        self.sd.load_controlnet_for_inpainting(
            'Lykon/DreamShaper',
            'lllyasviel/control_v11p_sd15_lineart'
        )

        self.masker.startup_model()

    def on_post_image(self):
        files = request.files
        image = files.get("image")

        img = Image.open(image) \
            .convert('RGB')
            #.convert("L") 
            
        img.save("zzz-posted-image.png")

        prompt = request.form.getlist('prompt')[0]
        print('prompt:   ', prompt)

        tensor_with_prediction = self.masker.preprocess_image(img)
        parsing_tensor = self.masker.parse_image(tensor_with_prediction)
        print('PARSING TENSOR SHAPE:   ', parsing_tensor.shape)
        mask = self.masker.generate_mask(parsing_tensor, [1, 17])
        PILmask = Image.fromarray(mask).convert('RGB')
        PILmask.save("zzz-mask.png")
        bbox = self.masker.get_second_opinion_as_bbox(img, 1, 0.6)   

        print('BBOX:   ', bbox)     


api = FlaskSDServer(
    CustomDiffuser('CPUExecutionProvider'),
    Masking(
        'cuda:0',#'cpu',
        BiSeNet(n_classes=19),
        load('C:/work/py/models/79999_iter.pth', device('cuda:0'))#'cpu'))
    )    
)
api.initialize_diffusor_and_masking()

@app.route('/diffusers/inpaint', methods=['POST'])
def post_prompt_for_inpaint():
    return api.on_post_image()
