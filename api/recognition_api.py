import sys
import argparse
import json
import time
from itertools import chain
from os.path import join as ospjoin
from typing import List

import cv2
from flask import request, send_file, flash, redirect

from detect.detecting_tool import UseOurDetect, make_byte_image_2_cv2, show_result_img
from recog.recog_tool import Recognition, for_same_name


def parse_args():
    parser = argparse.ArgumentParser(description='child_recognition')
    # general
    parser.add_argument('--rec_checkpoint_path', default=ospjoin('recog', '16_backbone.pth'),
                        help='path to load model.')
    parser.add_argument('--network', default='r100', type=str, help='')
    parser.add_argument("--feature_path", default="childbackup", type=str, help="if you wanna save image, put path")
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--batch-size', default=128, type=int, help='')
    parser.add_argument("--port", default=3333, type=int, help="port number")
    parser.add_argument("--save_image", default=ospjoin("api", "test"), type=str,
                        help="if you wanna save image, put path")
    parser.add_argument("--test", default="test", type=str, help=" test or None | test or 0")
    parser.add_argument("--who_r_u_threshold", default=38.5, type=float, help="")
    return parser.parse_args()


def request_check(class_name, args):
    if 'test' in class_name:
        class_name = class_name.replace('test', '')
        args.test = 'show_test'
    if 'high' in class_name:
        class_name = 'best'
        args.rec_checkpoint_path = ospjoin('recog', 'bestchild.pth')
    if 'speed' in class_name:
        class_name = class_name.replace('speed', '')
        args.test = 'speed'
    return class_name, args


def rec_predict(args):
    if not request.method == "POST":
        sys.exit()

    if request.files.getlist("image"):  # and len(request.files.getlist("image")) > 0:  # multi image
        image_files = request.files.getlist('image')
        if request.values.get('classname'):
            class_name = request.values.get('classname')
            class_name, args = request_check(class_name, args)
            args.feature_path = class_name
        results, total_name, total_result, all_score, times = dict(), list(), list(), list(), list()
        detect = UseOurDetect(args)
        recog = Recognition(args, detect.device)
        for file in image_files:
            name, result, scores, img, times = one_img_det_rec(file, recog, detect, args, times)
            total_name.append(name), total_result.append(result), all_score.append(scores)
            # save
            if args.save_img_path:
                save_path = for_same_name(args.save_img_path, name)
                cv2.imwrite(save_path, img)
                if args.test == 'test':
                    recog.write_json(result, name, args.test)

        if args.test == 'show_test':
            img_file = show_result_img(img)
            print(results)
            return send_file(img_file, mimetype='image/png')

        print('all of time : ', round(sum(times) / 1000, 3), 's | 1 image per time : ',
              round(sum(times) / len(image_files), 3), 'ms')
        total_result = who_r_u(all_score, total_result, args.who_r_u_threshold)
        results['image'], results['result'] = total_name, total_result

        results['speed'] = round(sum(times) / len(image_files), 3)
        print(all_score)
        box_num = len([i for i in chain(*all_score)])
        working_num = len([i for i in chain(*all_score) if i < args.who_r_u_threshold])
        working_per = 1 - working_num / box_num
        results['accuracy'] = working_per * 100
        print(working_per * 100, f'% work ==> this model thinks only {working_num}/{box_num} face have problem')
        return json.dumps(results)
    else:
        flash('No selected file')
        return redirect(request.url)


def one_img_det_rec(image_file, recog, detect, args, times):
    name = image_file.filename.split('.')[0]
    img = make_byte_image_2_cv2(image_file)
    start_time = time.time()
    # Detect
    file, crop_img, bbox = detect.detect_with_retinaface(img)
    if not args.test == 'speed':
        print("detect %d face | --- detect %s m seconds ---" % (len(bbox), round((time.time() - start_time) * 1000, 1)))
    if len(bbox) == 0:
        result, scores = [dict()], [0]
    # recognition
    else:
        class_id, result, img, scores, crop_img = recog.api_recognition(file, crop_img, bbox, img, args.test)
        if not args.test == 'speed':
            print("ID : %s | --- total %s m seconds ---" % (class_id, round((time.time() - start_time) * 1000, 1)))
        times.append((time.time() - start_time) * 1000)
        for i in range(len(crop_img)):
            save_path = ospjoin('api', 'test', f'{i}_crop_{bbox[0][0]}.jpg')
            cv2.imwrite(save_path, crop_img[i])

    return name, result, scores, img, times


def who_r_u(score: List[List[int]], result: List[List[dict]], threshold=38.5) -> List[List[dict]]:
    for z, score_cut in enumerate(score):
        for y, scc in enumerate(score_cut):
            if scc < threshold:
                result[z][y]['label'] = 'who are you?'
    return result
