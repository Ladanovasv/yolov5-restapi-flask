import argparse
import io
from PIL import Image
import os
import torch
from flask import Flask, request,  send_from_directory
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from yolov5.utils.plots import colors, plot_one_box
from yolov5.utils.torch_utils import select_device, load_classifier, time_sync
from yolov5.models.yolo import Model, Focus, Conv, C3, Bottleneck, SPP, Concat, Detect
import requests
from flask import jsonify
import json
import base64
from flask_cors import CORS

import uuid
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)

DETECTION_URL = "/"
CORS(app) 

app.config['UPLOADED_PATH'] = os.path.join(basedir, 'uploads')

@app.route('/files/<filename>')
def uploaded_files(filename):
    path = app.config['UPLOADED_PATH']
    return send_from_directory(path, filename)

def random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] 

@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return
    if request.data:
        data_json = request.data
        data_dict = json.loads(data_json)
        app.logger.info(data_dict['file']['uri'][:100])
        if(len(data_dict['file']['uri']) > 100):
            image_bytes = base64.b64decode(data_dict['file']['uri'][22:])
            img = Image.open(io.BytesIO(image_bytes))
        else:
            img = Image.open(requests.get(data_dict['file']['uri'], stream=True).raw)
            
        results = model(img, size=640)
            
        results.render()  # updates results.imgs with boxes and labels
        for img in results.imgs:
            img_base64 = Image.fromarray(img)
            uri = "image{}.jpg".format(random_string(10))
            img_base64.save("uploads/{}".format(uri), format="JPEG")

        data = results.pandas().xyxy[0].to_json(orient="records")

        response = {}
        response['data'] = json.loads(data)
        response['uri'] = 'http://127.0.0.1:5000/files/{}'.format(uri)
        # app.logger.info(response)
	
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    set_logging()
    device=''
    device = select_device(device)
   
    # Load model
    weights='weights/best.pt'
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, ['class{}'.format(i) for i in range(1000)]
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
    model = model.autoshape()
    model.eval()
    app.run(host="0.0.0.0", port=args.port,  debug=True)  # debug=True causes Restarting with stat
