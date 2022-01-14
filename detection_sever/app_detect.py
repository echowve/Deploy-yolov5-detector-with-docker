import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

from flask import Flask, make_response, jsonify
from flask_restful import abort, Api, Resource

import time, cv2, torch,json


from utils.tools import Parser
from PIL import Image
import io, numpy as np


from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load
from yolov5.make_detection import detect



class ObjectDetection():

    def __init__(self) -> None:

        self.device = select_device('0')
        self.model = attempt_load('yolov5s.pt', map_location=self.device)  # load FP32 model
        print('model_loaded')
        self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def predict(self, img):


        t1 = time.time()
        detections = detect(model=self.model, device=self.device, im0=img)
        t2 = time.time()
        time_pass = t2 -t1

        return detections, time_pass


detector = ObjectDetection()

app = Flask(__name__)
api = Api(app)
parser = Parser().getparser()






class detApi(Resource):

    def post(self):
        args = parser.parse_args()
        
        image_obj = args['files']
        shapex = args['shapex']
        shapey = args['shapey']
        shape = (shapex, shapey, 3)
        image = image_obj.read()

        img = np.frombuffer(image, dtype=np.uint8).reshape(shape)

        rep = detector.predict(img)
        return {"result":rep[0], 'time':rep[1]}




api.add_resource(detApi, '/detect')

@app.route('/welcome')
@app.route('/')
def Welcome():
   resp = make_response('<h1> Wecome Flask Rest API </h1><h1> Please set post to respose </h1>')
   return resp

if __name__ == '__main__':
    app.run(threaded=True, debug=False, host='0.0.0.0', port=5001)
