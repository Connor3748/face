import argparse
import json

from flask import Flask, request

from detect.detecting_tool import UseOurDetect, make_byte_image_2_PIL, draw_bbox_keypoint, write_csv

app = Flask(__name__)


def predict(args):
    if not request.method == "POST":
        return
    if request.files.getlist("image") and len(request.files.getlist("image")) > -1:  # multi image
        # start
        image_files = request.files.getlist('image')
        bboxes, results = list(), dict()
        detect = UseOurDetect(args)
        for file in image_files:
            try:
                img = make_byte_image_2_PIL(file)
            except:
                print('there are some wrong files. It not a image. please check file')
                return json.dumps(['there are some wrong files. It not a image. please check file'])
            re, _, bbox = detect.detect_with_retinaface(img_path=img)
            bboxes.append(bbox)
            if args.draw_bbox_keypoint:
                draw_bbox_keypoint(re, bbox, img)
            for box in bbox:
                write_csv(box, file.filename, args.save_img_path)
        results['bbox'] = bboxes

        return json.dumps(results)
