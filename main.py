from core.app import app
from waitress import serve
import json
with open('config/config.json', 'r') as f:
    configs = json.load(f)

if __name__ == '__main__':
    serve(app, host=configs['ip_address'], port=configs['port'])