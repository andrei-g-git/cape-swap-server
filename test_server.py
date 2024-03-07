#from _socket import _RetAddress
from http.server import HTTPServer, BaseHTTPRequestHandler
#from socketserver import BaseServer
import json
from custom_diffusers import CustomDiffuser

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print('received post request')

        self.send_response(200)
        self.path = '/diffuser'
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        length = int(self.headers.get("Content-length"))
        print('request lenght:   ' , length)
        body = json.loads(
            self.rfile
                .read(length)
                .decode('utf-8')
        )

        print('body:    \n', body)

        prompt = body['prompt']

        diffuser_handler = CustomDiffuser()

        diffuser_handler.load_model( #this shouldn't run every time
            path='./stable_diffusion_onnx',
            provider='CPU'
        )

        diffuser_handler.generate_text2image(
            prompt,
            '',
            512,
            512,
            10,
            7
        )

        diffuser_handler.save_image('plane.png')



HOST = 'localhost'
PORT = 9991

server = HTTPServer((HOST, PORT), RequestHandler)
print('server starting at ', HOST, ":", PORT)

server.serve_forever()
server.server_close()