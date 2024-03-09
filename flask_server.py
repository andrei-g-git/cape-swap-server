from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
import json
from custom_diffusers import CustomDiffuser

app = Flask(__name__)
CORS(app)

app.debug = True


# diffuser_handler = CustomDiffuser()
# diffuser_handler.load_model( #this shouldn't run every time
#     path='./stable_diffusion_onnx',
#     provider='CPU'
# )


class FlaskSDServer:
    def __init__(self, flask_app, diffusion_handler) -> None:
        self.diffusion_handler = diffusion_handler
        self.app = flask_app

    def start_server(self, HOST = 'localhost', PORT = 9991):

        self.diffusion_handler.load_model( #should be called separately...
            path='./stable_diffusion_onnx',
            provider='CPU'
        )


        if __name__ == '__main__':
            self.app.run(host=HOST, port=PORT)

    @app.route('/diffusers', methods=['POST'])
    def post_prompt():
        print('received POST request from flask server')

        body = json.loads(
            request.data.decode('utf-8'),
            strict=False
        )

        print('recieved body:    \n', body)

        prompt = body['prompt']

        response_data = {'simple_response': 'got prompt:    ' + prompt + '   hopefully next time I can send you a picture! '}
        response = make_response(jsonify(response_data))
        print('------------------\n', response, '\n------------------')
        print('jsonified data:   ', jsonify(response_data))
        response.headers['Content-Type'] = 'application/json'

        return response


server = FlaskSDServer(app, CustomDiffuser())
server.start_server('127.0.0.1', 5000)


# HOST = 'localhost'
# PORT = 9991


# if __name__ == '__main__':
#     app.run(host=HOST, port=PORT)
