import argparse
import os
from os.path import join as ospjoin

from flask import Flask, render_template, send_file

from api import recognition_api, detect_api, newclass_api

app = Flask(__name__)
Recognition_URL = ospjoin("/", "child", "recognition")
DETECTION_URL = ospjoin("/", "child", "detect")
Newclass_URL = ospjoin("/", "child", "newclass")
test = "/love"
test2 = "/our_team"

@app.route('/rec')
def index1():
    return render_template('recognition.html')


@app.route('/ann')
def index2():
    return render_template('detect.html')


@app.route('/new')
def index3():
    return render_template('newclass.html')


@app.route(Recognition_URL, methods=["POST"])
def service1():
    return recognition_api.rec_predict(args)


@app.route(DETECTION_URL, methods=["POST"])
def service2():
    return detect_api.predict(args)


@app.route(Newclass_URL, methods=["POST"])
def service3():
    return newclass_api.predict(args)


# TODO erase
@app.route(test)
def service4():
    import cv2, io, glob, random

    # image_paths = glob.glob(ospjoin('love', '*'))
    # image_path = random.choice(image_paths)
    # img = cv2.imread(image_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    # img = cv2.imencode('.png', img)[1].tostring()
    # f = io.BytesIO()
    # f.write(img)
    # f.seek(0)
    vedio_paths = glob.glob(ospjoin('love', '*.mp4'))
    return send_file(random.choice(vedio_paths), mimetype='video/mp4')
    # return send_file(f, mimetype='image/png')


@app.route(test2)
def service5():
    return send_file(ospjoin('love', 'KakaoTalk_20220516_150750200.png'), mimetype='image/png')


# TODO make same pathname
def parse_args():
    parser = argparse.ArgumentParser(description='child_recognition')
    # base
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--batch-size', default=128, type=int, help='')
    parser.add_argument("--port", default=3333, type=int, help="port number")
    parser.add_argument("--save_img_path", default="./api/test", type=str, help="if you wanna save image, put path")
    parser.add_argument("--test", default="speed", type=str, help="test or None | test or 0 or speed")
    # recog
    parser.add_argument('--rec_checkpoint_path', default='./recog/16_backbone.pth', help='path to load model.')
    parser.add_argument('--network', default='r100', type=str, help='')
    parser.add_argument("--feature_path", default="0104", type=str, help="if you wanna save image, put path")
    # detect
    parser.add_argument('-m', '--trained_detect_model', default='./detect/models/Resnet50_Final.pth', type=str,
                        help='Trained state_dict file path to open')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_threshold', default=0.6, type=float, help='visualization_threshold')
    parser.add_argument("--who_r_u_threshold", default=38.5, type=float, help="")
    parser.add_argument("--draw_bbox_keypoint", default='True', type=str, help="if you need result in face")


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_img_path, exist_ok=True)
    app.run(host="0.0.0.0", port=args.port)
