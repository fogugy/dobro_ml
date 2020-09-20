import json
import pickle
import warnings

from simple_http_server import request_map, server

from msg_classifier import MsgClassifier

warnings.filterwarnings("ignore")

with open('./predictor.model', 'rb') as f:
    proj_classifier = pickle.load(f)

msg_classifier = MsgClassifier()


@request_map("/project_type", method="POST")
def normal_form_post(text):
    r = proj_classifier.predict(text)
    return 200, json.dumps({'type': 'common' if r else 'personal'})


@request_map("/msg_score", method="POST")
def normal_form_post(text):
    r = msg_classifier.predict(text)
    return 200, json.dumps({'score': str(r[1])})


@request_map("/test", method="GET")
def test():
    return 200


server.start(port=4444)
