import gc
from io import BytesIO
import time
from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_cors import CORS, cross_origin
from PIL import Image
from torch import device, load, no_grad, cuda
import json
from custom_diffusers import CustomDiffuser
from image_processing import ImageProcessor
from masking import Masking
from bisnet import BiSeNet
import cv2
import numpy as np
import base64

class InterceptRequestMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        environ['Content-Type'] = 'image/png'
        return self.wsgi_app(environ, start_response)



app = Flask(__name__)
CORS(app)
#app.wsgi_app = InterceptRequestMiddleware(app.wsgi_app)
app.config['CORS_HEADERS'] = 'Content-Type'

app.debug = True

test_diffuser = CustomDiffuser('CUDAExecutionProvider')
test_masker = Masking(
        'cuda:0',
        BiSeNet(n_classes=19),
        load('C:/work/py/models/79999_iter.pth', device('cuda:0'))
    ) 




class FlaskSDServer:
    def __init__(self,  diffusion_handler: CustomDiffuser, masker: Masking, image_processor:ImageProcessor):
        self.sd = diffusion_handler
        self.masker = masker
        self.image_processor = image_processor
        self.inpaint_controlnet_pipe = None

    def initialize_diffusor_and_masking(self):
        self.inpaint_controlnet_pipe = self.sd.load_controlnet_for_inpainting(
            #'Lykon/DreamShaper',
            #'C:/work/py/models/imagepipeline/realisticVisionV60B1_v51VAE',
            #'imagepipeline/Realistic-Vision-V6.0',
            'rubbrband/realisticVisionV60B1_v60B1VAE',
            [
                'lllyasviel/control_v11p_sd15_lineart',
                'lllyasviel/control_v11p_sd15_normalbae'
            ]
            
        )
        self.masker.startup_model()
        self.image_processor.startup_normal_mapper()


    def on_post_image(self):
        #request.headers.set('Content-Type', 'image/png')

        files = request.files
        image = files.get("image")

        img = Image.open(image) \
            .convert('RGB')
            
        img.save("../images/out/zzz-posted-image.png")

        prompt = request.form.getlist('prompt')[0]
        print('prompt:   ', prompt)

        tensor_with_prediction = self.masker.preprocess_image(img)
        parsing_tensor = self.masker.parse_image(tensor_with_prediction)
        print('PARSING TENSOR SHAPE:   ', parsing_tensor.shape)
        mask = self.masker.generate_mask(parsing_tensor, [1, 17])
        PILmask = Image.fromarray(mask).convert('RGB')
        PILmask.save("../images/out/zzz-mask.png")
        bbox = self.masker.get_second_opinion_as_bbox(img, 1, 0.6)   
        print('BBOX:   ', bbox)  
        if len(bbox):
            head_and_hair_mask = self.masker.filter_contiguous_head(mask, bbox)
            selfie_mask = self.masker.mask_whole_person(np.asarray(img))
            cleaner_mask = self.masker.filter_biggest_segment(selfie_mask) #aren't I using this for the final product?...
            cv2.imwrite("../images/out/zzz-cleaner-mask.png", cleaner_mask)

            new_bbox = self.masker.get_bbox_from_mask(cleaner_mask)
            #image_path_no_extension = os.path.splitext(image_path)[0]
            headless_selfie_mask = self.masker.decapitate(selfie_mask, head_and_hair_mask)#cleaner_mask, head_and_hair_mask)

            control_image_for_preprocessing = np.asarray(img)
            canny_image = cv2.Canny(control_image_for_preprocessing, 100, 200)
            canny_image = Image.fromarray(canny_image)

            normal_map = self.image_processor.extract_normal_map(control_image_for_preprocessing)

            # final_input_image = self.masker.crop_image_subject_aware(img, new_bbox, 1.5)
            # final_mask = self.masker.crop_image_subject_aware(Image.fromarray(headless_selfie_mask.astype(np.uint8)), new_bbox, 1.5) # this isn't good I should get the padding only once
            # final_canny = self.masker.crop_image_subject_aware(canny_image, new_bbox, 1.5)
            # final_normal = self.masker.crop_image_subject_aware(normal_map, new_bbox, 1.5)


            final_input_image = self.image_processor.resize_image_keep_aspect(img, (768, 768)) 
            final_mask = self.image_processor.resize_image_keep_aspect(Image.fromarray(headless_selfie_mask.astype(np.uint8)), (768, 768))
            final_canny = self.image_processor.resize_image_keep_aspect(canny_image, (768, 768))
            final_normal = self.image_processor.resize_image_keep_aspect(normal_map, (768, 768))           

            new_w, new_h = final_input_image.size

            output = self.sd.inpaint_with_controlnet(
                final_input_image,
                final_mask,
                [final_canny, final_normal],
                768,
                512,               
                prompt                               
            )   


            w, h = final_input_image.size
            aspect_ratio = w/h
            resized_output = output.resize((w, h))
            
            resized_output.save("C:/work/py/cape-swap-server/images/out/zzz-generated.png", "PNG")        

            gc.collect()
            with no_grad():
                cuda.empty_cache()

            #im = Image.open("static/images/2.jpg")
            data = BytesIO()
            resized_output.save(data, "PNG")
            encoded_img_data = base64.b64encode(data.getvalue())

            return render_template("index.html", image_data=encoded_img_data.decode('utf-8'))

            # img_io = BytesIO() 
            # img.save(img_io, 'PNG', quality=75)
            # img_io.seek(0) #may not need
            # img_bytes_array = img_io.getvalue()

            # #self.inpaint_controlnet_pipe.to("cpu")

            # gc.collect()
            # with no_grad():
            #     cuda.empty_cache()


            # print('right before send file')

            # bytes_like_image = base64.b64encode(img_bytes_array).decode('utf-8')

            # #request.headers.set('Content-Type', 'image/png')
            # #return res_image#

            # #bytes_like_image = img_io.read()

            # #return bytes_like_image



            # time.sleep(2)
            # with open("C:/work/py/cape-swap-server/images/out/zzz-generated.png", "rb") as file:
            #     saved_img = file.read()
            #     saved_img_b64 = base64.b64encode(saved_img).decode('utf-8')


            #     response = {}
            #     response['data'] = saved_img_b64#bytes_like_image
            #     #return Response(json.dumps(response))#, mimetype='image/png')

            #     return saved_img_b64




            # #return send_file("C:/work/py/cape-swap-server/images/out/zzz-generated.png", mimetype='image/png')     



   


api = FlaskSDServer(
    CustomDiffuser('CUDAExecutionProvider'), 
    Masking(
        'cuda:0',#'cpu',
        BiSeNet(n_classes=19),
        load('C:/work/py/models/79999_iter.pth', device('cuda:0'))#'cpu'))
    ),
    ImageProcessor()    
)
api.initialize_diffusor_and_masking()

@app.route('/diffusers/inpaint', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def post_prompt_for_inpaint():
    return api.on_post_image()


@app.route('/test', methods=['GET'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def test_server():
    return 'hello this is a response from the server'


HOST = '127.0.0.1'
PORT = 5000


if __name__ == '__main__':

    app.run(host=HOST, port=PORT)
