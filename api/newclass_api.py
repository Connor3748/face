import argparse
import json
import os
import time
from os.path import join as ospjoin

import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

from detect.detecting_tool import UseOurDetect, make_byte_image_2_cv2
from recog.recog_tool import Recognition

app = Flask(__name__)


def predict(args):
    if not request.method == "POST":
        return
    if request.files.get("annotation"):
        class_name = request.values.get("classname")  # request.form["newclass"]
        js_file = request.files['annotation'].read().decode('utf-8')
        image_files = request.files.getlist("image")
        img_dir = img_path_setting(image_files, class_name)  # save image_file
        detect = UseOurDetect(args)
        start_time = time.time()
        # Detect Landmark + crop face
        file, crop_imgs, labels = detect.newcls_feature(img_dir, js_file)
        print("detect %d face | --- detect %s seconds ---" % (len(labels), (int(time.time() - start_time))))
        if len(crop_imgs) == 0:
            unique_templates = 'no face or json file have problem'
        # make feature / per class
        else:
            recog = Recognition(args, detect.device)
            feature, unique_templates, template2id = recog.newclass_feature(file, crop_imgs, labels)
            print("ID : %s | --- total %s seconds ---" % (unique_templates, (int(time.time() - start_time))))
            # save
            np.savez(ospjoin('api', 'newclass_backup', class_name, f'{class_name}_{len(labels)}_backup'),
                     feature=feature, unique_templates=unique_templates, template2id=template2id
                     )
            if args.test:
                class_id, scores = [], dict()
                for i in range(len(feature)):
                    score = np.sum(feature * feature[i][0], -1)
                    scores[i] = list(score)
                    predict = unique_templates[score.argmax()]
                    class_id.append(predict)
                print(pd.DataFrame.from_dict(scores))
        return json.dumps('Registration Number : ' + str(unique_templates))



def img_path_setting(image_files: list, classname: str) -> str:
    img_dir = ospjoin('api', 'newclass_backup', classname, 'temp')
    os.makedirs(img_dir, exist_ok=True)
    for file in image_files:
        name = file.filename
        img = make_byte_image_2_cv2(file)
        cv2.imwrite(ospjoin(img_dir, name), img)
    return img_dir
