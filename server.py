import json
import pickle
import warnings

from simple_http_server import request_map, server

warnings.filterwarnings("ignore")

with open('./predictor.model', 'rb') as f:
    p = pickle.load(f)


@request_map("/project_type", method="POST")
def normal_form_post(text):
    r = p.predict(text)
    return 200, json.dumps({'type': 'common' if r else 'personal'})


@request_map("/test", method="GET")
def test():
    return 200


server.start(port=4444)
