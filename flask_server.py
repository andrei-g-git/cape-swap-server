from flask import Flask, request, send_file
from flask_cors import CORS
import json
from custom_diffusers import CustomDiffuser
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)

app.debug = True


class FlaskSDServer:
    def __init__(self,  diffusion_handler):
        self.diffusion_handler = diffusion_handler


    def initialize_diffusor(self):
        self.diffusion_handler.load_model( #should be called separately...
            path='./stable_diffusion_onnx',
            provider='CPU'
        )

    def on_post_prompt(self):
        print('received POST request from flask server')

        body = json.loads(
            request.data.decode('utf-8'),
            strict=False
        )

        print('recieved body:    \n', body)

        prompt = body['prompt']

        image = self.diffusion_handler.generate_text2image(
            prompt,
            '',
            384,
            384,
            10,
            7
        )
        #self.diffusion_handler.save_image('generated_image.png')


        # response_data = {'simple_response': 'got prompt:    ' + prompt + '   hopefully next time I can send you a picture! '}
        # response = make_response(jsonify(response_data))
        # response.headers['Content-Type'] = 'application/json'


        img_io = BytesIO() 
        image.save(img_io, 'PNG', quality=75)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')



api = FlaskSDServer(CustomDiffuser())
api.initialize_diffusor()

@app.route('/diffusers', methods=['POST'])
def post_prompt():
    return api.on_post_prompt()


HOST = '127.0.0.1'
PORT = 5000


if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
