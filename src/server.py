from io import BytesIO
from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image
from torch import device, load
import json
from custom_diffusers import CustomDiffuser
from masking import Masking
from bisnet import BiSeNet
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

app.debug = True

test_diffuser = CustomDiffuser('CUDAExecutionProvider')
test_masker = Masking(
        'cuda:0',
        BiSeNet(n_classes=19),
        load('C:/work/py/models/79999_iter.pth', device('cuda:0'))
    ) 

class FlaskSDServer:
    def __init__(self,  diffusion_handler: CustomDiffuser, masker: Masking):
        self.sd = diffusion_handler
        self.masker = masker

    def initialize_diffusor_and_masking(self):
        self.sd.load_controlnet_for_inpainting(
            'Lykon/DreamShaper',
            #'C:/work/py/models/imagepipeline/realisticVisionV60B1_v51VAE',
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
        if len(bbox):
            head_and_hair_mask = self.masker.filter_contiguous_head(mask, bbox)
            selfie_mask = self.masker.mask_whole_person(np.asarray(img))
            cleaner_mask = self.masker.filter_biggest_segment(selfie_mask) #aren't I using this for the final product?...
            cv2.imwrite("zzz-cleaner-mask.png", cleaner_mask)

            new_bbox = self.masker.get_bbox_from_mask(cleaner_mask)
            #image_path_no_extension = os.path.splitext(image_path)[0]
            headless_selfie_mask = self.masker.decapitate(selfie_mask, head_and_hair_mask)#cleaner_mask, head_and_hair_mask)

            image_for_canny_edges = np.asarray(img)
            canny_image = cv2.Canny(image_for_canny_edges, 100, 200)
            canny_image = Image.fromarray(canny_image)

            final_input_image = self.masker.crop_image_subject_aware(img, new_bbox, 1.5)
            final_mask = self.masker.crop_image_subject_aware(Image.fromarray(headless_selfie_mask.astype(np.uint8)), new_bbox, 1.5) # this isn't good I should get the padding only once
            final_canny = self.masker.crop_image_subject_aware(canny_image, new_bbox, 1.5)

            new_w, new_h = final_input_image.size

            output = self.sd.inpaint_with_controlnet(
                final_input_image,
                final_mask,
                final_canny,
                768,
                512,               
                prompt                               
            )   


            w, h = final_input_image.size
            aspect_ratio = w/h
            resized_output = output.resize((w, h))
            
            resized_output.save("zzz-generated.png", "PNG")        

            img_io = BytesIO() 
            img.save(img_io, 'PNG', quality=75)
            img_io.seek(0)

            return send_file(img_io, mimetype='image/png')             
   


api = FlaskSDServer(
    CustomDiffuser('CUDAExecutionProvider'), 
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

HOST = '127.0.0.1'
PORT = 5000


if __name__ == '__main__':

    app.run(host=HOST, port=PORT)
