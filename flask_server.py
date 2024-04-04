from flask import Flask, request, send_file
from flask_cors import CORS
import json

from src.custom_diffusers import CustomDiffuser
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)

app.debug = True


class FlaskSDServer:
    def __init__(self,  diffusion_handler):
        self.diffusion_handler = diffusion_handler


    def initialize_diffusor(self):
        self.diffusion_handler.load_model_for_inpainting( #should be called separately...
            path='./stable_diffusion_onnx',
            #provider='CPU'
        )

    # def on_post_prompt(self):
    #     print('received POST request from flask server')

    #     body = json.loads(
    #         request.data.decode('utf-8'),
    #         strict=False
    #     )

    #     print('recieved body:    \n', body)

    #     prompt = body['prompt']

    #     image = self.diffusion_handler.generate_text2image(
    #         prompt,
    #         '',
    #         384,
    #         384,
    #         10,
    #         7
    #     )
    #     #self.diffusion_handler.save_image('generated_image.png')


    #     # response_data = {'simple_response': 'got prompt:    ' + prompt + '   hopefully next time I can send you a picture! '}
    #     # response = make_response(jsonify(response_data))
    #     # response.headers['Content-Type'] = 'application/json'


    #     img_io = BytesIO() 
    #     image.save(img_io, 'PNG', quality=75)
    #     img_io.seek(0)

    #     return send_file(img_io, mimetype='image/png')
    


    def on_post_image_for_inpaint(self):
        print('received POST request from flask server ---- IMAGE FOR INPAINT')

        #decoded_request = request.data.decode('utf-8')
        print("$$$$    request dict:    ", request.form.to_dict(flat=False))
        #print(decoded_request)
        #print("decoded request object type:    \n", type(decoded_request))

        #print("***** REQUEST DATA **********:    ", request.data)   # so there's a whole blob in there, don't know why request.files isn't seeing it...

        files = request.files
        image = files.get("image")

        print("image data type:   ", type(image))

        print("file list size", len(files))

        img = Image.open(image) \
            .convert("L") \
            .save("zzz-posted-image.png")





        # body = json.loads(
        #     request.data.decode('utf-8'),
        #     strict=False
        # )

        # print('recieved body:    \n', body)

        # prompt = body['prompt']
        # image = body['image']

        #blob = image.read()

        # print('IMAGE BLOB:   \n', blob)

        # # image = self.diffusion_handler.inpaint_with_prompt(

        # # )

        #return send_file({}, mimetype='image/png')
        return send_file({"foo": "bar"})


api = FlaskSDServer(CustomDiffuser('CPUExecutionProvider'))
#api.initialize_diffusor()

# @app.route('/diffusers', methods=['POST'])
# def post_prompt():
#     return api.on_post_prompt()

@app.route('/diffusers/inpaint', methods=['POST'])
def post_prompt_for_inpaint():
    return api.on_post_image_for_inpaint()


HOST = '127.0.0.1'
PORT = 5000


if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
