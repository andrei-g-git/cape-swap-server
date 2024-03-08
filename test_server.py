#from _socket import _RetAddress
from http.server import HTTPServer, BaseHTTPRequestHandler
import io
#from socketserver import BaseServer
import json
from socketserver import BaseServer
from custom_diffusers import CustomDiffuser
import asyncio

def make_request_handler(diffuser_handler):

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

            #diffuser_handler = CustomDiffuser()

            # self.diffuser_handler.load_model( #this shouldn't run every time
            #     path='./stable_diffusion_onnx',
            #     provider='CPU'
            # )

            image = diffuser_handler.generate_text2image(
                prompt,
                '',
                128,
                128,
                10,
                7
            )

            # image_bytes = io.BytesIO()
            # image.save(image_bytes, format='PNG')
            # image_bytes = image_bytes.getvalue()

            #self.wfile.write(image_bytes) #bytes(image)

            loop = asyncio.get_event_loop()
            loop.run_until_complete(lambda : diffuser_handler.save_image('generated_image.png')())
            loop.close()
            with open('generated_image.png', 'r') as image:
                self.wfile.write(bytes(image, 'image/png'))

            #diffuser_handler.save_image('plane.png')

    return RequestHandler

HOST = 'localhost'
PORT = 9991


diffuser_handler = CustomDiffuser()

diffuser_handler.load_model( #this shouldn't run every time
    path='./stable_diffusion_onnx',
    provider='CPU'
)

server = HTTPServer((HOST, PORT), make_request_handler(diffuser_handler))
print('server starting at ', HOST, ":", PORT)

server.serve_forever()
server.server_close()