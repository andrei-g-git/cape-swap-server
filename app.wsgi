import sys

sys.path.append('C:/work/py/cape-swap-server/src')

from src.server import app

app.secret_key = 'qwerty'